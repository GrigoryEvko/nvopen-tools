// Function: sub_A083B0
// Address: 0xa083b0
//
__int64 __fastcall sub_A083B0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 v6; // r12
  char v8; // al
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  bool v11; // cf
  __int64 v12; // r15
  __int64 *v13; // rdi
  __int64 v14; // r13
  __int64 v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // r8
  int v18; // edi
  unsigned int v19; // eax
  __int64 result; // rax
  __int64 v21; // rcx
  __int64 *v22; // rdi
  int v23; // r14d
  __int64 v24; // r13
  _QWORD *v25; // rax
  _QWORD *v26; // r13
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdi
  int v32; // eax
  __int64 v33; // r13
  __int64 v34; // r14
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 *v39; // r13
  __int64 *v40; // rdi
  int v41; // r12d
  int v42; // [rsp+0h] [rbp-70h]
  unsigned __int64 v43; // [rsp+0h] [rbp-70h]
  unsigned int v44; // [rsp+Ch] [rbp-64h] BYREF
  _DWORD v45[24]; // [rsp+10h] [rbp-60h] BYREF

  v6 = a2;
  v8 = *(_BYTE *)a2;
  v44 = a3;
  if ( (unsigned __int8)(v8 - 5) > 0x1Fu || (*(_BYTE *)(a2 + 1) & 0x7F) != 2 && !*(_DWORD *)(a2 - 8) )
  {
    v9 = *(unsigned int *)(a1 + 8);
    v10 = v44;
    v11 = v44 < (unsigned int)v9;
    if ( v44 != (_DWORD)v9 )
      goto LABEL_5;
LABEL_19:
    v21 = (unsigned int)v10;
    if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v10 )
    {
      v34 = a1 + 16;
      v39 = (__int64 *)sub_C8D7D0(a1, a1 + 16, 0, 8, v45);
      v40 = &v39[*(unsigned int *)(a1 + 8)];
      if ( v40 )
      {
        *v40 = v6;
        sub_B96E90(v40, v6, 1);
      }
      result = sub_A04E10(a1, v39, v35, v36, v37, v38);
      v41 = v45[0];
      if ( v34 != *(_QWORD *)a1 )
        result = _libc_free(*(_QWORD *)a1, v39);
      ++*(_DWORD *)(a1 + 8);
      *(_QWORD *)a1 = v39;
      *(_DWORD *)(a1 + 12) = v41;
    }
    else
    {
      result = *(_QWORD *)a1;
      v22 = (__int64 *)(*(_QWORD *)a1 + 8 * v21);
      if ( v22 )
      {
        *v22 = v6;
        result = sub_B96E90(v22, v6, 1);
        LODWORD(v9) = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v9 + 1;
    }
    return result;
  }
  a2 = a1 + 56;
  sub_A07210((__int64)v45, a1 + 56, (int *)&v44);
  v9 = *(unsigned int *)(a1 + 8);
  v10 = v44;
  v11 = v44 < (unsigned int)v9;
  if ( v44 == (_DWORD)v9 )
    goto LABEL_19;
LABEL_5:
  if ( v11 || (a6 = (unsigned int)(v10 + 1), v23 = v10 + 1, a6 == v9) )
  {
    v12 = *(_QWORD *)a1;
  }
  else
  {
    v24 = 8 * a6;
    if ( a6 < v9 )
    {
      v12 = *(_QWORD *)a1;
      v9 = *(_QWORD *)a1 + 8 * v9;
      v33 = *(_QWORD *)a1 + v24;
      if ( v9 != v33 )
      {
        do
        {
          a2 = *(_QWORD *)(v9 - 8);
          v9 -= 8LL;
          if ( a2 )
          {
            v43 = v9;
            sub_B91220(v9);
            v9 = v43;
          }
        }
        while ( v33 != v9 );
        v10 = v44;
        v12 = *(_QWORD *)a1;
      }
      *(_DWORD *)(a1 + 8) = v23;
    }
    else
    {
      if ( a6 > *(unsigned int *)(a1 + 12) )
      {
        a2 = sub_C8D7D0(a1, a1 + 16, a6, 8, v45);
        v12 = a2;
        sub_A04E10(a1, (__int64 *)a2, v27, v28, v29, v30);
        v31 = *(_QWORD *)a1;
        v32 = v45[0];
        if ( a1 + 16 != *(_QWORD *)a1 )
        {
          v42 = v45[0];
          _libc_free(v31, a2);
          v32 = v42;
        }
        *(_QWORD *)a1 = a2;
        v9 = *(unsigned int *)(a1 + 8);
        *(_DWORD *)(a1 + 12) = v32;
      }
      else
      {
        v12 = *(_QWORD *)a1;
      }
      v25 = (_QWORD *)(v12 + 8 * v9);
      v26 = (_QWORD *)(v12 + v24);
      if ( v25 != v26 )
      {
        do
        {
          if ( v25 )
            *v25 = 0;
          ++v25;
        }
        while ( v26 != v25 );
        v12 = *(_QWORD *)a1;
      }
      *(_DWORD *)(a1 + 8) = v23;
      v10 = v44;
    }
  }
  v13 = (__int64 *)(v12 + 8 * v10);
  v14 = *v13;
  if ( *v13 )
  {
    v15 = *(_QWORD *)(v14 + 8);
    if ( (v15 & 4) != 0 )
    {
      a2 = v6;
      sub_BA6110(v15 & 0xFFFFFFFFFFFFFFF8LL, v6);
      if ( (*(_BYTE *)(a1 + 32) & 1) == 0 )
        goto LABEL_10;
    }
    else if ( (*(_BYTE *)(a1 + 32) & 1) == 0 )
    {
LABEL_10:
      v16 = *(unsigned int *)(a1 + 48);
      v17 = *(_QWORD *)(a1 + 40);
      if ( !(_DWORD)v16 )
        return sub_BA65D0(v14, a2, v9, v16, v17, a6);
      v16 = (unsigned int)(v16 - 1);
LABEL_12:
      v9 = (unsigned int)v16 & (37 * v44);
      a2 = v17 + 4 * v9;
      v18 = *(_DWORD *)a2;
      if ( *(_DWORD *)a2 == v44 )
      {
LABEL_13:
        *(_DWORD *)a2 = -2;
        v19 = *(_DWORD *)(a1 + 32);
        ++*(_DWORD *)(a1 + 36);
        v9 = 2 * (v19 >> 1) - 2;
        *(_DWORD *)(a1 + 32) = v9 | v19 & 1;
      }
      else
      {
        a2 = 1;
        while ( v18 != -1 )
        {
          a6 = (unsigned int)(a2 + 1);
          v9 = (unsigned int)v16 & ((_DWORD)a2 + (_DWORD)v9);
          a2 = v17 + 4LL * (unsigned int)v9;
          v18 = *(_DWORD *)a2;
          if ( v44 == *(_DWORD *)a2 )
            goto LABEL_13;
          a2 = (unsigned int)a6;
        }
      }
      return sub_BA65D0(v14, a2, v9, v16, v17, a6);
    }
    v17 = a1 + 40;
    v16 = 0;
    goto LABEL_12;
  }
  *v13 = v6;
  return sub_B96E90(v13, v6, 1);
}
