// Function: sub_1F49DC0
// Address: 0x1f49dc0
//
__int64 __fastcall sub_1F49DC0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 (*v4)(void); // rax
  __int64 result; // rax
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rsi
  int v9; // edx

  v4 = (__int64 (*)(void))a2[5];
  if ( v4 )
  {
    result = v4();
    v6 = result;
    v8 = v7;
  }
  else
  {
    result = *a2;
    v8 = *(unsigned __int16 *)(*a2 + 20);
    v6 = *(_QWORD *)result;
  }
  if ( v8 )
  {
    v9 = 0;
    result = 0;
    do
    {
      *(_QWORD *)(*a3 + (((unsigned __int64)*(unsigned __int16 *)(v6 + 2 * result) >> 3) & 0x1FF8)) |= 1LL << *(_WORD *)(v6 + 2 * result);
      result = (unsigned int)++v9;
    }
    while ( v9 != v8 );
  }
  return result;
}
