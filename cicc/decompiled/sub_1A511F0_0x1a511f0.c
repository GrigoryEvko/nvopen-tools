// Function: sub_1A511F0
// Address: 0x1a511f0
//
__int64 *__fastcall sub_1A511F0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // eax
  __int64 v6; // rcx
  __int64 *v7; // rbx
  __int64 v8; // rdx
  __int64 *v9; // r8
  __int64 *v10; // r15
  __int64 v11; // r14
  int v12; // r8d
  int v13; // r9d
  unsigned __int8 v14; // al
  __int64 *v15; // rax
  char v16; // dl
  __int64 v17; // rax
  unsigned __int64 v19; // r9
  __int64 v20; // rax
  unsigned __int64 v21; // r9
  __int64 v22; // rdx
  unsigned __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 *v26; // rsi
  __int64 *v27; // rcx
  unsigned __int64 v28; // [rsp+8h] [rbp-F8h]
  unsigned __int64 v29; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v30; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v31; // [rsp+20h] [rbp-E0h]
  _QWORD *v32; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v33; // [rsp+38h] [rbp-C8h]
  _QWORD v34[4]; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v35; // [rsp+60h] [rbp-A0h] BYREF
  __int64 *v36; // [rsp+68h] [rbp-98h]
  __int64 *v37; // [rsp+70h] [rbp-90h]
  __int64 v38; // [rsp+78h] [rbp-88h]
  int v39; // [rsp+80h] [rbp-80h]
  _BYTE v40[120]; // [rsp+88h] [rbp-78h] BYREF

  *a1 = 0;
  v32 = v34;
  v36 = (__int64 *)v40;
  v37 = (__int64 *)v40;
  v35 = 0;
  v38 = 8;
  v39 = 0;
  v34[0] = a3;
  v33 = 0x400000001LL;
  sub_1412190((__int64)&v35, a3);
  v5 = 1;
  do
  {
    v6 = v5--;
    v7 = (__int64 *)v32[v6 - 1];
    LODWORD(v33) = v5;
    v8 = 24LL * (*((_DWORD *)v7 + 5) & 0xFFFFFFF);
    if ( (*((_BYTE *)v7 + 23) & 0x40) != 0 )
    {
      v9 = (__int64 *)*(v7 - 1);
      v7 = &v9[(unsigned __int64)v8 / 8];
    }
    else
    {
      v9 = &v7[v8 / 0xFFFFFFFFFFFFFFF8LL];
    }
    if ( v9 != v7 )
    {
      v10 = v9;
      while ( 1 )
      {
        v11 = *v10;
        if ( *(_BYTE *)(*v10 + 16) <= 0x10u )
          goto LABEL_6;
        if ( sub_13FC1A0(a2, *v10) )
        {
          v19 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v19 )
          {
            if ( (*a1 & 4) == 0 )
            {
              v29 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
              v20 = sub_22077B0(48);
              v21 = v29;
              if ( v20 )
              {
                *(_QWORD *)v20 = v20 + 16;
                *(_QWORD *)(v20 + 8) = 0x400000000LL;
              }
              v22 = v20;
              v23 = v20 & 0xFFFFFFFFFFFFFFF8LL;
              *a1 = v22 | 4;
              v24 = *(unsigned int *)(v23 + 8);
              if ( (unsigned int)v24 >= *(_DWORD *)(v23 + 12) )
              {
                v28 = v29;
                v31 = v23;
                sub_16CD150(v23, (const void *)(v23 + 16), 0, 8, v12, v21);
                v23 = v31;
                v21 = v28;
                v24 = *(unsigned int *)(v31 + 8);
              }
              *(_QWORD *)(*(_QWORD *)v23 + 8 * v24) = v21;
              ++*(_DWORD *)(v23 + 8);
              v19 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
            }
            v25 = *(unsigned int *)(v19 + 8);
            if ( (unsigned int)v25 >= *(_DWORD *)(v19 + 12) )
            {
              v30 = v19;
              sub_16CD150(v19, (const void *)(v19 + 16), 0, 8, v12, v19);
              v19 = v30;
              v25 = *(unsigned int *)(v30 + 8);
            }
            *(_QWORD *)(*(_QWORD *)v19 + 8 * v25) = v11;
            ++*(_DWORD *)(v19 + 8);
          }
          else
          {
            *a1 = v11;
          }
          goto LABEL_6;
        }
        v14 = *(_BYTE *)(v11 + 16);
        if ( v14 <= 0x17u || v14 != *(_BYTE *)(a3 + 16) )
          goto LABEL_6;
        v15 = v36;
        if ( v37 == v36 )
        {
          v26 = &v36[HIDWORD(v38)];
          if ( v36 != v26 )
          {
            v27 = 0;
            while ( v11 != *v15 )
            {
              if ( *v15 == -2 )
                v27 = v15;
              if ( v26 == ++v15 )
              {
                if ( !v27 )
                  goto LABEL_43;
                *v27 = v11;
                v17 = (unsigned int)v33;
                --v39;
                ++v35;
                if ( (unsigned int)v33 < HIDWORD(v33) )
                  goto LABEL_14;
                goto LABEL_41;
              }
            }
            goto LABEL_6;
          }
LABEL_43:
          if ( HIDWORD(v38) < (unsigned int)v38 )
          {
            ++HIDWORD(v38);
            *v26 = v11;
            ++v35;
            goto LABEL_13;
          }
        }
        sub_16CCBA0((__int64)&v35, v11);
        if ( v16 )
        {
LABEL_13:
          v17 = (unsigned int)v33;
          if ( (unsigned int)v33 >= HIDWORD(v33) )
          {
LABEL_41:
            sub_16CD150((__int64)&v32, v34, 0, 8, v12, v13);
            v17 = (unsigned int)v33;
          }
LABEL_14:
          v10 += 3;
          v32[v17] = v11;
          LODWORD(v33) = v33 + 1;
          if ( v7 == v10 )
          {
LABEL_15:
            v5 = v33;
            break;
          }
        }
        else
        {
LABEL_6:
          v10 += 3;
          if ( v7 == v10 )
            goto LABEL_15;
        }
      }
    }
  }
  while ( v5 );
  if ( v37 != v36 )
    _libc_free((unsigned __int64)v37);
  if ( v32 != v34 )
    _libc_free((unsigned __int64)v32);
  return a1;
}
