// Function: sub_1A21230
// Address: 0x1a21230
//
char __fastcall sub_1A21230(__int64 *a1, __int64 *a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  unsigned __int64 v14; // r8
  unsigned __int64 v15; // rcx
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  _QWORD *v19; // r8
  int v20; // ecx
  _QWORD *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rbx
  _QWORD *v24; // rdi
  unsigned __int8 v25; // al
  __int64 v26; // rbx
  __int64 v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rdx
  char v30; // si
  __int64 v31; // rax
  __int64 v33; // [rsp+0h] [rbp-40h]
  int v34; // [rsp+8h] [rbp-38h]
  __int64 v35; // [rsp+8h] [rbp-38h]

  v10 = *a1;
  v11 = *a2;
  if ( *a1 > (unsigned __int64)*a2 )
  {
    v12 = 0;
  }
  else
  {
    v12 = (v11 - v10) / a4;
    if ( v12 * a4 != v11 - v10 )
      goto LABEL_3;
  }
  v14 = *(_QWORD *)(a3 + 32);
  if ( v14 > v12 )
  {
    v15 = a2[1];
    if ( a1[1] <= v15 )
      v15 = a1[1];
    v16 = v15 - v10;
    v17 = v16 / a4;
    if ( v16 / a4 * a4 == v16 && v17 <= v14 )
    {
      v18 = v17 - v12;
      v19 = *(_QWORD **)(a3 + 24);
      v20 = v18;
      if ( v18 != 1 )
      {
        v34 = v18;
        v21 = sub_16463B0(*(__int64 **)(a3 + 24), v18);
        v20 = v34;
        v19 = v21;
      }
      v35 = (__int64)v19;
      v22 = sub_1644C60(*(_QWORD **)a3, 8 * (int)a4 * v20);
      v23 = a2[2];
      v33 = v22;
      v24 = sub_1648700(v23 & 0xFFFFFFFFFFFFFFF8LL);
      v25 = *((_BYTE *)v24 + 16);
      if ( v25 > 0x17u )
      {
        if ( v25 == 78 )
        {
          v29 = *(v24 - 3);
          if ( !*(_BYTE *)(v29 + 16) )
          {
            v30 = *(_BYTE *)(v29 + 33);
            if ( (v30 & 0x20) != 0 )
            {
              if ( (unsigned int)(*(_DWORD *)(v29 + 36) - 133) <= 4
                && ((1LL << (*(_BYTE *)(v29 + 36) + 123)) & 0x15) != 0 )
              {
                if ( !(unsigned __int8)sub_1A211D0((__int64)v24) )
                  return (v23 >> 2) & 1;
              }
              else if ( (v30 & 0x20) != 0 )
              {
                LOBYTE(v13) = (unsigned int)(*(_DWORD *)(v29 + 36) - 116) <= 1;
                return v13;
              }
            }
          }
          goto LABEL_3;
        }
        if ( *(_BYTE *)(**(_QWORD **)(**(_QWORD **)(v23 & 0xFFFFFFFFFFFFFFF8LL) + 16LL) + 8LL) != 13 )
        {
          if ( v25 == 54 )
          {
            if ( (*((_BYTE *)v24 + 18) & 1) != 0 )
              goto LABEL_3;
            if ( *a2 >= (unsigned __int64)*a1 )
            {
              v31 = v33;
              if ( a2[1] <= (unsigned __int64)a1[1] )
                v31 = *v24;
              v33 = v31;
            }
            v28 = v33;
            v27 = v35;
          }
          else
          {
            if ( v25 != 55 || (*((_BYTE *)v24 + 18) & 1) != 0 )
              goto LABEL_3;
            if ( *a2 >= (unsigned __int64)*a1 )
            {
              v26 = v33;
              if ( a2[1] <= (unsigned __int64)a1[1] )
                v26 = *(_QWORD *)*(v24 - 6);
              v33 = v26;
            }
            v27 = v33;
            v28 = v35;
          }
          LOBYTE(v13) = sub_1A1E350(a5, v27, v28);
          return v13;
        }
      }
    }
  }
LABEL_3:
  LOBYTE(v13) = 0;
  return v13;
}
