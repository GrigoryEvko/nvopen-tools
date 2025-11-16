// Function: sub_2B232B0
// Address: 0x2b232b0
//
bool __fastcall sub_2B232B0(_QWORD **a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rsi
  _BYTE *v6; // r8
  bool result; // al
  __int64 *v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 *v11; // rbx

  v4 = *a2;
  v6 = *(_BYTE **)(v4 + 416);
  if ( *(_DWORD *)(v4 + 104) != 3 )
    goto LABEL_2;
  if ( v6 && *(_QWORD *)(v4 + 424) && *v6 == 90 )
  {
LABEL_4:
    result = 1;
    if ( *v6 != 91 )
    {
      result = 0;
      if ( *v6 == 84 )
      {
        v11 = (__int64 *)(*(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 8));
        return v11 == sub_2B22F90(*(__int64 **)v4, (__int64)v11, *a1, a4);
      }
    }
    return result;
  }
  v8 = *(__int64 **)v4;
  v9 = *(_QWORD *)v4 + 8LL * *(unsigned int *)(v4 + 8);
  if ( *(_QWORD *)v4 == v9 )
    return 1;
  v10 = 0;
  do
  {
    LOBYTE(a4) = *(_BYTE *)*v8++ == 90;
    a4 = (unsigned __int8)a4;
    v10 += (unsigned __int8)a4;
  }
  while ( (__int64 *)v9 != v8 );
  result = 1;
  if ( v10 > 4 )
  {
LABEL_2:
    result = 0;
    if ( !v6 || !*(_QWORD *)(v4 + 424) )
      return result;
    goto LABEL_4;
  }
  return result;
}
