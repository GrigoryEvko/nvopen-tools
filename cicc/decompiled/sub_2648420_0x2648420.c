// Function: sub_2648420
// Address: 0x2648420
//
bool __fastcall sub_2648420(_QWORD *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rbp
  __int64 v4; // rsi
  bool result; // al
  __int64 v6; // rcx
  __int64 v7; // rax
  unsigned int v8; // ebx
  _QWORD v10[4]; // [rsp-58h] [rbp-58h] BYREF
  _QWORD v11[7]; // [rsp-38h] [rbp-38h] BYREF

  v4 = *a2;
  result = 0;
  if ( *(_DWORD *)(v4 + 40) )
  {
    result = 1;
    if ( *(_DWORD *)(*(_QWORD *)a3 + 40LL) )
    {
      v6 = *(unsigned __int8 *)(v4 + 16);
      v7 = *(unsigned __int8 *)(*(_QWORD *)a3 + 16LL);
      if ( (_BYTE)v6 == (_BYTE)v7 )
      {
        v11[6] = v3;
        sub_22B0690(v10, (__int64 *)(v4 + 24));
        v8 = *(_DWORD *)v10[2];
        sub_22B0690(v11, (__int64 *)(*(_QWORD *)a3 + 24LL));
        return v8 < *(_DWORD *)v11[2];
      }
      else
      {
        return *(_DWORD *)(*a1 + 4 * v6) < *(_DWORD *)(*a1 + 4 * v7);
      }
    }
  }
  return result;
}
