// Function: sub_2AC7A10
// Address: 0x2ac7a10
//
unsigned __int64 __fastcall sub_2AC7A10(__int64 a1, __int64 a2, __int64 a3)
{
  bool v3; // zf
  __int64 v4; // rax
  unsigned __int64 result; // rax
  int v6; // esi
  int v7; // edx
  unsigned int v8; // esi
  __int64 v9; // rdx
  unsigned __int8 **v10; // rdx
  __int64 v11; // r14
  __int64 v12; // rax
  bool v13; // of
  __int64 v14; // [rsp+10h] [rbp-50h] BYREF
  __int64 v15; // [rsp+18h] [rbp-48h] BYREF
  __int64 v16; // [rsp+20h] [rbp-40h] BYREF
  int v17; // [rsp+28h] [rbp-38h]
  char v18; // [rsp+2Ch] [rbp-34h]

  if ( BYTE4(a3) || (_DWORD)a3 != 1 )
  {
    v16 = a2;
    v17 = a3;
    v18 = BYTE4(a3);
    v3 = (unsigned __int8)sub_2ABE410(a1 + 352, &v16, &v14) == 0;
    v4 = v14;
    if ( !v3 )
      return *(_QWORD *)(v14 + 24);
    v6 = *(_DWORD *)(a1 + 368);
    ++*(_QWORD *)(a1 + 352);
    v15 = v4;
    v7 = v6 + 1;
    v8 = *(_DWORD *)(a1 + 376);
    if ( 4 * v7 >= 3 * v8 )
    {
      v8 *= 2;
    }
    else if ( v8 - *(_DWORD *)(a1 + 372) - v7 > v8 >> 3 )
    {
      goto LABEL_7;
    }
    sub_2AC77A0(a1 + 352, v8);
    sub_2ABE410(a1 + 352, &v16, &v15);
    v7 = *(_DWORD *)(a1 + 368) + 1;
    v4 = v15;
LABEL_7:
    *(_DWORD *)(a1 + 368) = v7;
    if ( *(_QWORD *)v4 != -4096 || *(_DWORD *)(v4 + 8) != -1 || !*(_BYTE *)(v4 + 12) )
      --*(_DWORD *)(a1 + 372);
    v9 = v16;
    *(_QWORD *)(v4 + 24) = 0;
    *(_DWORD *)(v4 + 32) = 0;
    *(_QWORD *)v4 = v9;
    *(_DWORD *)(v4 + 8) = v17;
    LOBYTE(v9) = v18;
    *(_DWORD *)(v4 + 16) = 0;
    *(_BYTE *)(v4 + 12) = v9;
    return 0;
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v10 = *(unsigned __int8 ***)(a2 - 8);
  else
    v10 = (unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  sub_DFB770(*v10);
  v11 = sub_DFD4A0(*(__int64 **)(a1 + 448));
  v12 = sub_DFDB90(*(_QWORD *)(a1 + 448));
  v13 = __OFADD__(v11, v12);
  result = v11 + v12;
  if ( v13 )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v11 <= 0 )
      return 0x8000000000000000LL;
  }
  return result;
}
