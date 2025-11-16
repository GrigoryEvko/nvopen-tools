// Function: sub_1E0FB20
// Address: 0x1e0fb20
//
__int64 __fastcall sub_1E0FB20(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 v7; // r15
  __int64 v8; // rbx
  unsigned int v9; // eax
  _QWORD *v10; // r9
  __int64 v11; // r8
  _QWORD *v12; // rbx
  _QWORD *v13; // r12
  __int64 result; // rax
  __int64 v15; // rdi
  __int64 v16; // r8
  unsigned int v17; // eax
  __int64 v18; // rcx
  int v19; // esi
  int v20; // r11d
  _QWORD *v21; // rcx
  int v22; // eax
  unsigned int v23; // edx
  __int64 v24; // r9
  int v25; // edi
  _QWORD *v26; // rsi
  int v27; // edi
  unsigned int v28; // edx
  __int64 v29; // r8
  __int64 v30; // [rsp+10h] [rbp-50h] BYREF
  __int64 v31; // [rsp+18h] [rbp-48h]
  __int64 v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+28h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_QWORD *)(a1 + 16);
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v4 = (v3 - v2) >> 4;
  v33 = 0;
  if ( (_DWORD)v4 )
  {
    v5 = 0;
    v6 = 0;
    v7 = 16LL * (unsigned int)(v4 - 1);
    while ( 1 )
    {
      v8 = v5 + v2;
      if ( *(int *)(v8 + 8) >= 0 )
      {
        if ( v5 == v7 )
          goto LABEL_11;
        goto LABEL_4;
      }
      if ( (_DWORD)v33 )
      {
        v9 = (v33 - 1) & (((unsigned int)*(_QWORD *)v8 >> 9) ^ ((unsigned int)*(_QWORD *)v8 >> 4));
        v10 = (_QWORD *)(v6 + 8LL * v9);
        v11 = *v10;
        if ( *(_QWORD *)v8 == *v10 )
          goto LABEL_8;
        v20 = 1;
        v21 = 0;
        while ( v11 != -8 )
        {
          if ( !v21 && v11 == -16 )
            v21 = v10;
          v9 = (v33 - 1) & (v20 + v9);
          v10 = (_QWORD *)(v6 + 8LL * v9);
          v11 = *v10;
          if ( *(_QWORD *)v8 == *v10 )
            goto LABEL_8;
          ++v20;
        }
        if ( !v21 )
          v21 = v10;
        ++v30;
        v22 = v32 + 1;
        if ( 4 * ((int)v32 + 1) < (unsigned int)(3 * v33) )
        {
          if ( (int)v33 - HIDWORD(v32) - v22 > (unsigned int)v33 >> 3 )
            goto LABEL_40;
          sub_1E0F6A0((__int64)&v30, v33);
          if ( !(_DWORD)v33 )
          {
LABEL_70:
            LODWORD(v32) = v32 + 1;
            BUG();
          }
          v27 = 1;
          v26 = 0;
          v28 = (v33 - 1) & (((unsigned int)*(_QWORD *)v8 >> 9) ^ ((unsigned int)*(_QWORD *)v8 >> 4));
          v21 = (_QWORD *)(v31 + 8LL * v28);
          v29 = *v21;
          v22 = v32 + 1;
          if ( *v21 == *(_QWORD *)v8 )
            goto LABEL_40;
          while ( v29 != -8 )
          {
            if ( v29 == -16 && !v26 )
              v26 = v21;
            v28 = (v33 - 1) & (v27 + v28);
            v21 = (_QWORD *)(v31 + 8LL * v28);
            v29 = *v21;
            if ( *(_QWORD *)v8 == *v21 )
              goto LABEL_40;
            ++v27;
          }
          goto LABEL_49;
        }
      }
      else
      {
        ++v30;
      }
      sub_1E0F6A0((__int64)&v30, 2 * v33);
      if ( !(_DWORD)v33 )
        goto LABEL_70;
      v23 = (v33 - 1) & (((unsigned int)*(_QWORD *)v8 >> 9) ^ ((unsigned int)*(_QWORD *)v8 >> 4));
      v21 = (_QWORD *)(v31 + 8LL * v23);
      v24 = *v21;
      v22 = v32 + 1;
      if ( *(_QWORD *)v8 == *v21 )
        goto LABEL_40;
      v25 = 1;
      v26 = 0;
      while ( v24 != -8 )
      {
        if ( v24 == -16 && !v26 )
          v26 = v21;
        v23 = (v33 - 1) & (v25 + v23);
        v21 = (_QWORD *)(v31 + 8LL * v23);
        v24 = *v21;
        if ( *(_QWORD *)v8 == *v21 )
          goto LABEL_40;
        ++v25;
      }
LABEL_49:
      if ( v26 )
        v21 = v26;
LABEL_40:
      LODWORD(v32) = v22;
      if ( *v21 != -8 )
        --HIDWORD(v32);
      *v21 = *(_QWORD *)v8;
      v11 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + v5);
LABEL_8:
      if ( v11 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 16LL))(v11);
      v6 = v31;
      if ( v5 == v7 )
        goto LABEL_11;
LABEL_4:
      v2 = *(_QWORD *)(a1 + 8);
      v5 += 16;
    }
  }
  v6 = 0;
LABEL_11:
  v12 = *(_QWORD **)(a1 + 40);
  v13 = &v12[*(unsigned int *)(a1 + 56)];
  if ( *(_DWORD *)(a1 + 48) && v13 != v12 )
  {
    while ( *v12 == -8 || *v12 == -16 )
    {
      if ( v13 == ++v12 )
        goto LABEL_12;
    }
    while ( v12 != v13 )
    {
      v16 = *v12;
      if ( (_DWORD)v33 )
      {
        v17 = (v33 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v18 = *(_QWORD *)(v6 + 8LL * v17);
        if ( v16 == v18 )
          goto LABEL_23;
        v19 = 1;
        while ( v18 != -8 )
        {
          v17 = (v33 - 1) & (v19 + v17);
          v18 = *(_QWORD *)(v6 + 8LL * v17);
          if ( v16 == v18 )
            goto LABEL_23;
          ++v19;
        }
      }
      if ( v16 )
      {
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v16 + 16LL))(*v12);
        v6 = v31;
      }
LABEL_23:
      if ( ++v12 == v13 )
        break;
      while ( *v12 == -16 || *v12 == -8 )
      {
        if ( v13 == ++v12 )
          goto LABEL_12;
      }
    }
  }
LABEL_12:
  j___libc_free_0(v6);
  result = j___libc_free_0(*(_QWORD *)(a1 + 40));
  v15 = *(_QWORD *)(a1 + 8);
  if ( v15 )
    return j_j___libc_free_0(v15, *(_QWORD *)(a1 + 24) - v15);
  return result;
}
