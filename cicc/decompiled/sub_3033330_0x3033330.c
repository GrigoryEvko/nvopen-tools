// Function: sub_3033330
// Address: 0x3033330
//
__int64 __fastcall sub_3033330(__int64 a1, unsigned int a2, _DWORD *a3, unsigned int a4)
{
  int v5; // eax
  unsigned int v6; // r12d
  __int64 v8; // rax
  unsigned __int16 v9; // dx
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  unsigned __int16 v13; // dx
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int16 v16; // [rsp+0h] [rbp-40h] BYREF
  __int64 v17; // [rsp+8h] [rbp-38h]
  unsigned __int16 v18; // [rsp+10h] [rbp-30h] BYREF
  __int64 v19; // [rsp+18h] [rbp-28h]

  *a3 = 2;
  v5 = *(_DWORD *)(a1 + 24);
  LOBYTE(a4) = v5 == 213 || v5 == 222;
  v6 = a4;
  if ( !(_BYTE)a4 )
  {
    if ( v5 != 214 )
      return v6;
    v12 = *(_QWORD *)(**(_QWORD **)(a1 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a1 + 40) + 8LL);
    v13 = *(_WORD *)v12;
    v14 = *(_QWORD *)(v12 + 8);
    v16 = v13;
    v17 = v14;
    if ( !v13 )
    {
      v15 = sub_3007260((__int64)&v16);
      goto LABEL_10;
    }
    if ( v13 != 1 && (unsigned __int16)(v13 - 504) > 7u )
    {
      v15 = *(_QWORD *)&byte_444C4A0[16 * v13 - 16];
LABEL_10:
      if ( a2 >= v15 )
      {
        *a3 = 1;
        return 1;
      }
      return v6;
    }
LABEL_19:
    BUG();
  }
  v8 = *(_QWORD *)(**(_QWORD **)(a1 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a1 + 40) + 8LL);
  v9 = *(_WORD *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v18 = v9;
  v19 = v10;
  if ( v9 )
  {
    if ( v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
      goto LABEL_19;
    v11 = *(_QWORD *)&byte_444C4A0[16 * v9 - 16];
  }
  else
  {
    v11 = sub_3007260((__int64)&v18);
  }
  if ( v11 > a2 )
    return 0;
  *a3 = 0;
  return v6;
}
