// Function: sub_19C2F80
// Address: 0x19c2f80
//
__int64 __fastcall sub_19C2F80(_BYTE *a1, __int64 a2, _BYTE *a3, unsigned int *a4, __int64 a5)
{
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned int v11; // ecx
  _QWORD *v12; // rdx
  _BYTE *v13; // r8
  __int64 v14; // r8
  int v16; // edx
  __int64 v17; // r9
  char v18; // al
  unsigned int v19; // ebx
  __int64 v20; // rax
  _QWORD *v21; // rax
  unsigned int v22; // r8d
  __int64 v23; // rsi
  __int64 v24; // r9
  unsigned int v25; // eax
  __int64 *v26; // rdx
  __int64 v27; // rcx
  int v28; // r10d
  _QWORD *v29; // rax
  int v30; // r11d
  __int64 *v31; // rdi
  int v32; // eax
  int v33; // edi
  int v34; // eax
  int v35; // ecx
  __int64 v36; // r9
  unsigned int v37; // eax
  __int64 v38; // r8
  int v39; // r11d
  __int64 *v40; // r10
  int v41; // eax
  int v42; // ecx
  __int64 v43; // r9
  __int64 *v44; // r10
  int v45; // r11d
  unsigned int v46; // eax
  __int64 v47; // [rsp+0h] [rbp-40h]
  __int64 v48; // [rsp+0h] [rbp-40h]
  __int64 v49[7]; // [rsp+8h] [rbp-38h] BYREF

  v9 = *(unsigned int *)(a5 + 24);
  v49[0] = (__int64)a1;
  if ( !(_DWORD)v9 )
    goto LABEL_8;
  v10 = *(_QWORD *)(a5 + 8);
  v11 = (v9 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v12 = (_QWORD *)(v10 + 16LL * v11);
  v13 = (_BYTE *)*v12;
  if ( a1 != (_BYTE *)*v12 )
  {
    v16 = 1;
    while ( v13 != (_BYTE *)-8LL )
    {
      v28 = v16 + 1;
      v11 = (v9 - 1) & (v16 + v11);
      v12 = (_QWORD *)(v10 + 16LL * v11);
      v13 = (_BYTE *)*v12;
      if ( a1 == (_BYTE *)*v12 )
        goto LABEL_3;
      v16 = v28;
    }
LABEL_8:
    if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 16 || a1[16] <= 0x10u )
      return 0;
    if ( (unsigned __int8)sub_13FC6C0(a2, (__int64)a1, a3, 0) )
    {
      v29 = sub_19C2D40(a5, v49);
      v14 = v49[0];
      v29[1] = v49[0];
      return v14;
    }
    v17 = v49[0];
    v18 = *(_BYTE *)(v49[0] + 16);
    if ( (unsigned __int8)(v18 - 50) <= 1u )
    {
      v19 = *a4;
      if ( *a4 == 2 )
      {
        if ( v18 != 50 )
          goto LABEL_21;
      }
      else if ( v19 > 2 )
      {
        if ( v19 == 3 )
          goto LABEL_21;
      }
      else if ( v19 )
      {
        v19 = 1;
        if ( v18 != 51 )
          goto LABEL_21;
      }
      else
      {
        v19 = (v18 == 50) + 1;
      }
      *a4 = v19;
      v47 = v17;
      v20 = sub_19C2F80(*(_QWORD *)(v17 - 48), a2, a3, a4, a5);
      if ( v20 )
      {
        v48 = v20;
        goto LABEL_18;
      }
      *a4 = v19;
      v48 = sub_19C2F80(*(_QWORD *)(v47 - 24), a2, a3, a4, a5);
      if ( v48 )
      {
LABEL_18:
        v21 = sub_19C2D40(a5, v49);
        v14 = v48;
        v21[1] = v48;
        return v14;
      }
    }
LABEL_21:
    v22 = *(_DWORD *)(a5 + 24);
    if ( v22 )
    {
      v23 = v49[0];
      v24 = *(_QWORD *)(a5 + 8);
      v25 = (v22 - 1) & ((LODWORD(v49[0]) >> 9) ^ (LODWORD(v49[0]) >> 4));
      v26 = (__int64 *)(v24 + 16LL * v25);
      v27 = *v26;
      if ( *v26 == v49[0] )
      {
LABEL_23:
        v26[1] = 0;
        return 0;
      }
      v30 = 1;
      v31 = 0;
      while ( v27 != -8 )
      {
        if ( v27 == -16 && !v31 )
          v31 = v26;
        v25 = (v22 - 1) & (v30 + v25);
        v26 = (__int64 *)(v24 + 16LL * v25);
        v27 = *v26;
        if ( v49[0] == *v26 )
          goto LABEL_23;
        ++v30;
      }
      v32 = *(_DWORD *)(a5 + 16);
      if ( v31 )
        v26 = v31;
      ++*(_QWORD *)a5;
      v33 = v32 + 1;
      if ( 4 * (v32 + 1) < 3 * v22 )
      {
        if ( v22 - *(_DWORD *)(a5 + 20) - v33 > v22 >> 3 )
        {
LABEL_34:
          *(_DWORD *)(a5 + 16) = v33;
          if ( *v26 != -8 )
            --*(_DWORD *)(a5 + 20);
          *v26 = v23;
          v26[1] = 0;
          goto LABEL_23;
        }
        sub_176F940(a5, v22);
        v41 = *(_DWORD *)(a5 + 24);
        if ( v41 )
        {
          v42 = v41 - 1;
          v43 = *(_QWORD *)(a5 + 8);
          v44 = 0;
          v45 = 1;
          v33 = *(_DWORD *)(a5 + 16) + 1;
          v46 = (v41 - 1) & ((LODWORD(v49[0]) >> 9) ^ (LODWORD(v49[0]) >> 4));
          v26 = (__int64 *)(v43 + 16LL * v46);
          v23 = *v26;
          if ( v49[0] != *v26 )
          {
            while ( v23 != -8 )
            {
              if ( !v44 && v23 == -16 )
                v44 = v26;
              v46 = v42 & (v45 + v46);
              v26 = (__int64 *)(v43 + 16LL * v46);
              v23 = *v26;
              if ( v49[0] == *v26 )
                goto LABEL_34;
              ++v45;
            }
            v23 = v49[0];
            if ( v44 )
              v26 = v44;
          }
          goto LABEL_34;
        }
LABEL_69:
        ++*(_DWORD *)(a5 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a5;
    }
    sub_176F940(a5, 2 * v22);
    v34 = *(_DWORD *)(a5 + 24);
    if ( v34 )
    {
      v23 = v49[0];
      v35 = v34 - 1;
      v36 = *(_QWORD *)(a5 + 8);
      v33 = *(_DWORD *)(a5 + 16) + 1;
      v37 = (v34 - 1) & ((LODWORD(v49[0]) >> 9) ^ (LODWORD(v49[0]) >> 4));
      v26 = (__int64 *)(v36 + 16LL * v37);
      v38 = *v26;
      if ( *v26 != v49[0] )
      {
        v39 = 1;
        v40 = 0;
        while ( v38 != -8 )
        {
          if ( v38 == -16 && !v40 )
            v40 = v26;
          v37 = v35 & (v39 + v37);
          v26 = (__int64 *)(v36 + 16LL * v37);
          v38 = *v26;
          if ( v49[0] == *v26 )
            goto LABEL_34;
          ++v39;
        }
        if ( v40 )
          v26 = v40;
      }
      goto LABEL_34;
    }
    goto LABEL_69;
  }
LABEL_3:
  if ( v12 == (_QWORD *)(v10 + 16 * v9) )
    goto LABEL_8;
  return v12[1];
}
