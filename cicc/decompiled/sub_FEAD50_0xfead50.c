// Function: sub_FEAD50
// Address: 0xfead50
//
char __fastcall sub_FEAD50(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  unsigned int v4; // r14d
  __int64 v5; // rax
  unsigned __int64 v6; // rbx
  __int64 v7; // rax
  _DWORD *v8; // rdi
  __int64 v9; // r11
  unsigned __int64 *v10; // rdx
  __int64 v11; // r15
  unsigned int v12; // edx
  unsigned __int64 v13; // rax
  bool v14; // cf
  unsigned __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 *v17; // r9
  unsigned __int64 *v18; // rdx
  __int64 *v20; // [rsp+0h] [rbp-70h]
  __int64 *v21; // [rsp+0h] [rbp-70h]
  unsigned __int64 v22; // [rsp+8h] [rbp-68h]
  unsigned __int64 v23; // [rsp+8h] [rbp-68h]
  __int64 v24; // [rsp+10h] [rbp-60h]
  __int64 v25; // [rsp+10h] [rbp-60h]
  __int64 v26; // [rsp+28h] [rbp-48h]
  unsigned int v27; // [rsp+38h] [rbp-38h] BYREF
  unsigned int v28[13]; // [rsp+3Ch] [rbp-34h] BYREF

  sub_FE9FC0(a2);
  v3 = *(_QWORD *)a2;
  v4 = *(_DWORD *)(a2 + 80);
  v5 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  v26 = v5;
  if ( v5 != *(_QWORD *)a2 )
  {
    v6 = -1;
    while ( 1 )
    {
      v11 = *(_QWORD *)(v3 + 8);
      v12 = v4;
      v4 -= v11;
      sub_F02DB0(v28, v11, v12);
      v27 = v28[0];
      v13 = sub_F02E20(&v27, v6);
      v14 = v6 < v13;
      v6 -= v13;
      v15 = v13;
      if ( v14 )
        v6 = 0;
      v5 = *(_QWORD *)(a1 + 64);
      v16 = v5 + 24LL * *(unsigned int *)(v3 + 4);
      v17 = *(__int64 **)(v16 + 8);
      if ( !v17 )
        goto LABEL_13;
      v7 = *((unsigned int *)v17 + 3);
      v8 = (_DWORD *)v17[12];
      if ( (unsigned int)v7 > 1 )
      {
        v20 = *(__int64 **)(v16 + 8);
        v22 = v15;
        v24 = v16;
        LOBYTE(v5) = sub_FDC990(v8, &v8[v7], (_DWORD *)v16);
        v16 = v24;
        v15 = v22;
        v17 = v20;
        if ( !(_BYTE)v5 )
        {
          v18 = (unsigned __int64 *)(v24 + 16);
          goto LABEL_14;
        }
LABEL_5:
        if ( !*((_BYTE *)v17 + 8) )
          goto LABEL_13;
        v9 = *v17;
        if ( !*v17 )
          goto LABEL_8;
        v5 = *(unsigned int *)(v9 + 12);
        if ( (unsigned int)v5 <= 1
          || (v21 = v17,
              v23 = v15,
              v25 = *v17,
              LOBYTE(v5) = sub_FDC990(*(_DWORD **)(v9 + 96), (_DWORD *)(*(_QWORD *)(v9 + 96) + 4 * v5), (_DWORD *)v16),
              v15 = v23,
              v17 = v21,
              !(_BYTE)v5)
          || (v10 = (unsigned __int64 *)(v25 + 152), !*(_BYTE *)(v25 + 8)) )
        {
LABEL_8:
          v10 = (unsigned __int64 *)(v17 + 19);
        }
        *v10 = v15;
        v3 += 16;
        if ( v3 == v26 )
          return v5;
      }
      else
      {
        LODWORD(v5) = *v8;
        if ( *(_DWORD *)v16 == *v8 )
          goto LABEL_5;
LABEL_13:
        v18 = (unsigned __int64 *)(v16 + 16);
LABEL_14:
        *v18 = v15;
        v3 += 16;
        if ( v3 == v26 )
          return v5;
      }
    }
  }
  return v5;
}
