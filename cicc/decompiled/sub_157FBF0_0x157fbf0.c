// Function: sub_157FBF0
// Address: 0x157fbf0
//
__int64 __fastcall sub_157FBF0(_QWORD *a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // r15
  _QWORD *v8; // rax
  __int64 v9; // r12
  __int64 v10; // rsi
  _QWORD *v11; // r14
  unsigned __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // r13
  _QWORD *v16; // r14
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r14
  __int64 v23; // rsi
  int v24; // r8d
  unsigned int v25; // edi
  char v26; // r10
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rcx
  __int64 v31; // rcx
  unsigned int v32; // edi
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v37; // rsi
  int v39; // [rsp+0h] [rbp-50h]
  __int64 v40; // [rsp+8h] [rbp-48h]
  unsigned int i; // [rsp+8h] [rbp-48h]
  __int64 v42; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v43[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = a1[4];
  if ( !v5 || v5 == a1[7] + 72LL )
    v6 = 0;
  else
    v6 = v5 - 24;
  v40 = a1[7];
  v7 = sub_157E9C0((__int64)a1);
  v8 = (_QWORD *)sub_22077B0(64);
  v9 = (__int64)v8;
  if ( v8 )
    sub_157FB60(v8, v7, a3, v40, v6);
  if ( !a2 )
    BUG();
  v10 = a2[3];
  v42 = v10;
  if ( v10 )
    sub_1623A60(&v42, v10, 2);
  v11 = a1 + 5;
  if ( a1 + 5 != a2 && (_QWORD *)(v9 + 40) != v11 )
  {
    sub_157EA80(v9 + 40, (__int64)(a1 + 5), (__int64)a2, (__int64)(a1 + 5));
    v12 = a1[5] & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v11;
    v13 = *(_QWORD *)(v9 + 40);
    a1[5] = a1[5] & 7LL | *a2 & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v12 + 8) = v9 + 40;
    v13 &= 0xFFFFFFFFFFFFFFF8LL;
    *a2 = v13 | *a2 & 7;
    *(_QWORD *)(v13 + 8) = a2;
    *(_QWORD *)(v9 + 40) = v12 | *(_QWORD *)(v9 + 40) & 7LL;
  }
  v14 = sub_1648A60(56, 1);
  v15 = v14;
  if ( v14 )
    sub_15F8590(v14, v9, a1);
  v16 = (_QWORD *)(v15 + 48);
  v43[0] = v42;
  if ( !v42 )
  {
    if ( v16 == v43 || !*(_QWORD *)(v15 + 48) )
      goto LABEL_18;
LABEL_47:
    sub_161E7C0(v15 + 48);
    goto LABEL_48;
  }
  sub_1623A60(v43, v42, 2);
  if ( v16 == v43 )
  {
    if ( v43[0] )
      sub_161E7C0(v15 + 48);
    goto LABEL_18;
  }
  if ( *(_QWORD *)(v15 + 48) )
    goto LABEL_47;
LABEL_48:
  v37 = v43[0];
  *(_QWORD *)(v15 + 48) = v43[0];
  if ( v37 )
    sub_1623210(v43, v37, v15 + 48);
LABEL_18:
  v17 = sub_157EBA0(v9);
  v18 = v17;
  if ( v17 )
  {
    v39 = sub_15F4D60(v17);
    if ( v39 )
    {
      for ( i = 0; i != v39; ++i )
      {
        v19 = sub_15F4DF0(v18, i);
        v20 = sub_157F280(v19);
        v22 = v21;
        v23 = v20;
        while ( v22 != v23 )
        {
          while ( 1 )
          {
            v24 = *(_DWORD *)(v23 + 20);
            v25 = v24 & 0xFFFFFFF;
            if ( (v24 & 0xFFFFFFF) != 0 )
            {
              v26 = *(_BYTE *)(v23 + 23);
              v27 = *(unsigned int *)(v23 + 56);
              v28 = 24 * v27 + 8;
              v29 = 0;
              while ( 1 )
              {
                v30 = v23 - 24LL * v25;
                if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
                  v30 = *(_QWORD *)(v23 - 8);
                if ( a1 == *(_QWORD **)(v30 + v28) )
                  break;
                v29 = (unsigned int)(v29 + 1);
                v28 += 8;
                if ( v25 == (_DWORD)v29 )
                  goto LABEL_37;
              }
              while ( 1 )
              {
                v31 = (v26 & 0x40) != 0 ? *(_QWORD *)(v23 - 8) : v23 - 24LL * (v24 & 0xFFFFFFF);
                *(_QWORD *)(v31 + 8 * v29 + 24 * v27 + 8) = v9;
                v24 = *(_DWORD *)(v23 + 20);
                v32 = v24 & 0xFFFFFFF;
                if ( (v24 & 0xFFFFFFF) == 0 )
                  break;
                v26 = *(_BYTE *)(v23 + 23);
                v27 = *(unsigned int *)(v23 + 56);
                v33 = 24 * v27 + 8;
                v29 = 0;
                while ( 1 )
                {
                  v34 = v23 - 24LL * v32;
                  if ( (*(_BYTE *)(v23 + 23) & 0x40) != 0 )
                    v34 = *(_QWORD *)(v23 - 8);
                  if ( a1 == *(_QWORD **)(v34 + v33) )
                    break;
                  v29 = (unsigned int)(v29 + 1);
                  v33 += 8;
                  if ( v32 == (_DWORD)v29 )
                    goto LABEL_37;
                }
              }
            }
LABEL_37:
            v35 = *(_QWORD *)(v23 + 32);
            if ( !v35 )
              BUG();
            v23 = 0;
            if ( *(_BYTE *)(v35 - 8) != 77 )
              break;
            v23 = v35 - 24;
            if ( v22 == v35 - 24 )
              goto LABEL_40;
          }
        }
LABEL_40:
        ;
      }
    }
  }
  if ( v42 )
    sub_161E7C0(&v42);
  return v9;
}
