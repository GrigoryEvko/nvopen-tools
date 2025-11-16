// Function: sub_1B8FA60
// Address: 0x1b8fa60
//
__int64 __fastcall sub_1B8FA60(__int64 a1, unsigned int a2, __int64 *a3)
{
  __int64 *v4; // rax
  __int64 v5; // r13
  unsigned int v6; // r13d
  char v7; // al
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 *v11; // rbx
  __int64 v12; // r10
  unsigned __int64 v13; // r9
  unsigned __int64 v14; // r15
  _BYTE *v15; // rax
  int v16; // edx
  __int64 v17; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r15
  int v22; // r15d
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  int v26; // edx
  _QWORD *v27; // r15
  __int64 v28; // rdx
  __int64 v29; // rax
  _QWORD *v30; // rbx
  _QWORD *v31; // rax
  __int64 v32; // [rsp+8h] [rbp-78h]
  unsigned __int64 v33; // [rsp+10h] [rbp-70h]
  unsigned __int64 v34; // [rsp+10h] [rbp-70h]
  _BYTE *v35; // [rsp+20h] [rbp-60h] BYREF
  __int64 v36; // [rsp+28h] [rbp-58h]
  _BYTE v37[80]; // [rsp+30h] [rbp-50h] BYREF

  v4 = sub_1B8E090(*(__int64 **)a1, a2);
  if ( !*((_BYTE *)v4 + 8) || (v5 = (__int64)v4, *(_BYTE *)(a1 + 16) == 54) && (unsigned __int8)sub_14A2EA0((__int64)a3) )
  {
    v7 = *(_BYTE *)(a1 + 16);
    v6 = 0;
    if ( v7 == 78 )
    {
LABEL_17:
      if ( *(char *)(a1 + 23) < 0 )
      {
        v19 = sub_1648A40(a1);
        v21 = v19 + v20;
        if ( *(char *)(a1 + 23) >= 0 )
        {
          if ( (unsigned int)(v21 >> 4) )
            goto LABEL_39;
        }
        else if ( (unsigned int)((v21 - sub_1648A40(a1)) >> 4) )
        {
          if ( *(char *)(a1 + 23) < 0 )
          {
            v22 = *(_DWORD *)(sub_1648A40(a1) + 8);
            if ( *(char *)(a1 + 23) >= 0 )
              BUG();
            v23 = sub_1648A40(a1);
            v25 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v23 + v24 - 4) - v22);
            goto LABEL_25;
          }
LABEL_39:
          BUG();
        }
      }
      v25 = -24;
LABEL_25:
      v26 = *(_DWORD *)(a1 + 20);
      v27 = (_QWORD *)(a1 + v25);
      v35 = v37;
      v36 = 0x400000000LL;
      v28 = 24LL * (v26 & 0xFFFFFFF);
      v29 = v28 + v25;
      v30 = (_QWORD *)(a1 - v28);
      v13 = 0xAAAAAAAAAAAAAAABLL * (v29 >> 3);
      if ( (unsigned __int64)v29 > 0x60 )
      {
        v33 = 0xAAAAAAAAAAAAAAABLL * (v29 >> 3);
        sub_16CD150((__int64)&v35, v37, v33, 8, (int)v37, v13);
        v16 = v36;
        LODWORD(v13) = v33;
        v31 = &v35[8 * (unsigned int)v36];
      }
      else
      {
        v31 = v37;
        v16 = 0;
      }
      if ( v27 == v30 )
        goto LABEL_12;
      do
      {
        if ( v31 )
          *v31 = *v30;
        v30 += 3;
        ++v31;
      }
      while ( v27 != v30 );
      goto LABEL_11;
    }
  }
  else
  {
    v6 = sub_14A2E40(a3, v5, 1u, 0);
    v7 = *(_BYTE *)(a1 + 16);
    if ( v7 == 78 )
      goto LABEL_17;
  }
  if ( v7 == 55 && (unsigned __int8)sub_14A2EA0((__int64)a3) )
    return v6;
  v8 = sub_13CF970(a1);
  v9 = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v10 = v8 + v9;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v11 = *(__int64 **)(a1 - 8);
  else
    v11 = (__int64 *)(a1 - v9);
  v12 = v10 - (_QWORD)v11;
  v35 = v37;
  v36 = 0x400000000LL;
  v13 = 0xAAAAAAAAAAAAAAABLL * ((v10 - (__int64)v11) >> 3);
  v14 = v13;
  if ( (unsigned __int64)(v10 - (_QWORD)v11) > 0x60 )
  {
    v32 = v10 - (_QWORD)v11;
    v34 = 0xAAAAAAAAAAAAAAABLL * (v12 >> 3);
    sub_16CD150((__int64)&v35, v37, v34, 8, (int)v37, v13);
    v16 = v36;
    LODWORD(v13) = v34;
    v12 = v32;
    v15 = &v35[8 * (unsigned int)v36];
  }
  else
  {
    v15 = v37;
    v16 = 0;
  }
  if ( v12 > 0 )
  {
    do
    {
      v17 = *v11;
      v15 += 8;
      v11 += 3;
      *((_QWORD *)v15 - 1) = v17;
      --v14;
    }
    while ( v14 );
LABEL_11:
    v16 = v36;
  }
LABEL_12:
  LODWORD(v36) = v13 + v16;
  v6 += sub_14A2E70((__int64)a3);
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
  return v6;
}
