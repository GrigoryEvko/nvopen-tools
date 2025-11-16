// Function: sub_121A060
// Address: 0x121a060
//
__int64 __fastcall sub_121A060(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  int v4; // eax
  unsigned __int64 v5; // rax
  __int64 v6; // r8
  unsigned int v7; // r9d
  __int64 v9; // rax
  __int64 *v10; // r9
  __int64 *v11; // rdi
  __int64 v12; // r9
  int v13; // eax
  unsigned __int64 v14; // rax
  char v15; // al
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 *v18; // r10
  unsigned __int64 v19; // [rsp+10h] [rbp-80h]
  __int64 *v20; // [rsp+10h] [rbp-80h]
  unsigned __int8 v21; // [rsp+10h] [rbp-80h]
  unsigned __int64 v22; // [rsp+18h] [rbp-78h]
  unsigned __int8 v23; // [rsp+18h] [rbp-78h]
  __int64 *v24; // [rsp+18h] [rbp-78h]
  __int64 *v25; // [rsp+28h] [rbp-68h] BYREF
  int v26[8]; // [rsp+30h] [rbp-60h] BYREF
  char v27; // [rsp+50h] [rbp-40h]
  char v28; // [rsp+51h] [rbp-3Fh]

  v3 = a1 + 176;
  v4 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v4;
  if ( v4 == 9 )
  {
    v13 = sub_1205200(v3);
    v7 = 0;
    *(_DWORD *)(a1 + 240) = v13;
    return v7;
  }
  v5 = *(_QWORD *)(a1 + 232);
  v25 = 0;
  v22 = v5;
  v28 = 1;
  *(_QWORD *)v26 = "expected type";
  v27 = 3;
  if ( !(unsigned __int8)sub_12190A0(a1, &v25, v26, 0) )
  {
    v9 = *(unsigned int *)(a2 + 8);
    v10 = v25;
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      v20 = v25;
      sub_C8D5F0(a2, (const void *)(a2 + 16), v9 + 1, 8u, v6, (__int64)v25);
      v9 = *(unsigned int *)(a2 + 8);
      v10 = v20;
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v9) = v10;
    v11 = v25;
    ++*(_DWORD *)(a2 + 8);
    LOBYTE(v12) = sub_BCBA50((__int64)v11);
    if ( (_BYTE)v12 )
    {
      if ( *(_DWORD *)(a1 + 240) != 4 )
        return (unsigned int)sub_120AFE0(a1, 9, "expected '}' at end of struct");
      while ( 1 )
      {
        v23 = v12;
        *(_DWORD *)(a1 + 240) = sub_1205200(v3);
        v14 = *(_QWORD *)(a1 + 232);
        v28 = 1;
        v19 = v14;
        *(_QWORD *)v26 = "expected type";
        v27 = 3;
        if ( (unsigned __int8)sub_12190A0(a1, &v25, v26, 0) )
          break;
        v15 = sub_BCBA50((__int64)v25);
        v12 = v23;
        if ( !v15 )
        {
          v28 = 1;
          *(_QWORD *)v26 = "invalid element type for struct";
          v27 = 3;
          sub_11FD800(v3, v19, (__int64)v26, 1);
          return v23;
        }
        v17 = *(unsigned int *)(a2 + 8);
        v18 = v25;
        if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          v21 = v23;
          v24 = v25;
          sub_C8D5F0(a2, (const void *)(a2 + 16), v17 + 1, 8u, v16, v12);
          v17 = *(unsigned int *)(a2 + 8);
          LOBYTE(v12) = v21;
          v18 = v24;
        }
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v17) = v18;
        ++*(_DWORD *)(a2 + 8);
        if ( *(_DWORD *)(a1 + 240) != 4 )
          return (unsigned int)sub_120AFE0(a1, 9, "expected '}' at end of struct");
      }
    }
    else
    {
      v28 = 1;
      *(_QWORD *)v26 = "invalid element type for struct";
      v27 = 3;
      sub_11FD800(v3, v22, (__int64)v26, 1);
    }
  }
  return 1;
}
