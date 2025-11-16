// Function: sub_D0E9D0
// Address: 0xd0e9d0
//
__int64 __fastcall sub_D0E9D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rsi
  unsigned int v9; // eax
  unsigned int v10; // edx
  __int64 v11; // rcx
  __int64 v12; // rsi
  unsigned int v13; // eax
  unsigned int v14; // r15d
  unsigned int v16; // eax
  __int64 v17; // rdx
  unsigned int v18; // eax
  bool v19; // al
  __int64 v20; // rdx
  unsigned int v21; // eax
  __int64 v22; // [rsp+8h] [rbp-148h]
  _QWORD v23[2]; // [rsp+10h] [rbp-140h] BYREF
  _QWORD v24[38]; // [rsp+20h] [rbp-130h] BYREF

  if ( !a4 )
  {
LABEL_12:
    v23[0] = v24;
    v24[0] = a1;
    v23[1] = 0x2000000001LL;
    v14 = sub_D0E9A0((__int64)v23, a2, a3, a4, a5, (__int64)v24);
    if ( (_QWORD *)v23[0] != v24 )
      _libc_free(v23[0], a2);
    return v14;
  }
  if ( a1 )
  {
    v8 = (unsigned int)(*(_DWORD *)(a1 + 44) + 1);
    v9 = *(_DWORD *)(a1 + 44) + 1;
  }
  else
  {
    v8 = 0;
    v9 = 0;
  }
  v10 = *(_DWORD *)(a4 + 32);
  if ( v9 < v10 )
  {
    v11 = *(_QWORD *)(a4 + 24);
    if ( *(_QWORD *)(v11 + 8 * v8) )
    {
      if ( a2 )
      {
        v12 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
        v13 = *(_DWORD *)(a2 + 44) + 1;
      }
      else
      {
        v12 = 0;
        v13 = 0;
      }
      if ( v10 <= v13 || !*(_QWORD *)(v11 + 8 * v12) )
        return 0;
    }
  }
  if ( a3 && *(_DWORD *)(a3 + 24) != *(_DWORD *)(a3 + 20) )
    goto LABEL_12;
  v22 = a5;
  LOBYTE(v16) = sub_AA5B70(a1);
  v14 = v16;
  if ( !(_BYTE)v16
    || (!a2 ? (v17 = 0, v18 = 0) : (v17 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1), v18 = *(_DWORD *)(a2 + 44) + 1),
        v18 >= *(_DWORD *)(a4 + 32) || !*(_QWORD *)(*(_QWORD *)(a4 + 24) + 8 * v17)) )
  {
    v19 = sub_AA5B70(a2);
    a5 = v22;
    if ( !v19 )
      goto LABEL_12;
    if ( a1 )
    {
      v20 = (unsigned int)(*(_DWORD *)(a1 + 44) + 1);
      v21 = *(_DWORD *)(a1 + 44) + 1;
    }
    else
    {
      v20 = 0;
      v21 = 0;
    }
    if ( v21 >= *(_DWORD *)(a4 + 32) || !*(_QWORD *)(*(_QWORD *)(a4 + 24) + 8 * v20) )
      goto LABEL_12;
    return 0;
  }
  return v14;
}
