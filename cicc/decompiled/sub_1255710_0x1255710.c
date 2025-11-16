// Function: sub_1255710
// Address: 0x1255710
//
__int64 __fastcall sub_1255710(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 result; // rax
  __int64 v7; // r12
  volatile signed __int32 *v8; // rdx
  __int64 v9; // rax

  result = sub_22077B0(80);
  if ( !result )
  {
    *(_QWORD *)(a1 + 32) = a3;
    *(_QWORD *)a1 = 16;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 16;
    *(_QWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 40) = 1;
    return result;
  }
  v7 = result;
  *(_QWORD *)(result + 8) = 0x100000001LL;
  v8 = (volatile signed __int32 *)(result + 8);
  *(_QWORD *)result = off_497C168;
  *(_QWORD *)(result + 16) = off_49E6938;
  *(_DWORD *)(result + 56) = a4;
  *(_QWORD *)(result + 32) = a2;
  *(_QWORD *)(result + 24) = &unk_49E68E0;
  *(_QWORD *)(result + 40) = a3;
  *(_QWORD *)(result + 64) = a2;
  *(_QWORD *)(result + 48) = &unk_49E6828;
  v9 = result + 16;
  *(_QWORD *)(v7 + 72) = a3;
  *(_QWORD *)a1 = v9;
  *(_QWORD *)(a1 + 8) = v7;
  if ( &_pthread_key_create )
    _InterlockedAdd(v8, 1u);
  else
    ++*(_DWORD *)(v7 + 8);
  *(_QWORD *)(a1 + 16) = v9;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = a3;
  *(_BYTE *)(a1 + 40) = 1;
  if ( &_pthread_key_create )
  {
    result = (unsigned int)_InterlockedExchangeAdd(v8, 0xFFFFFFFF);
    if ( (_DWORD)result != 1 )
      return result;
  }
  else
  {
    result = *(unsigned int *)(v7 + 8);
    *(_DWORD *)(v7 + 8) = result - 1;
    if ( (_DWORD)result != 1 )
      return result;
  }
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 16LL))(v7);
  if ( &_pthread_key_create )
  {
    result = (unsigned int)_InterlockedExchangeAdd((volatile signed __int32 *)(v7 + 12), 0xFFFFFFFF);
  }
  else
  {
    result = *(unsigned int *)(v7 + 12);
    *(_DWORD *)(v7 + 12) = result - 1;
  }
  if ( (_DWORD)result == 1 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 24LL))(v7);
  return result;
}
