// Function: sub_2FF5520
// Address: 0x2ff5520
//
__int64 __fastcall sub_2FF5520(__int64 a1, unsigned __int16 ***a2, _QWORD *a3)
{
  __int64 (*v4)(void); // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r8
  unsigned __int16 *v8; // rdx
  __int64 result; // rax
  unsigned __int16 *i; // rsi
  unsigned __int64 v11; // rcx

  v4 = (__int64 (*)(void))a2[8];
  if ( v4 )
  {
    v5 = v4();
    v7 = v6;
    v8 = (unsigned __int16 *)v5;
    result = v7;
  }
  else
  {
    result = *((unsigned __int16 *)*a2 + 10);
    v8 = **a2;
  }
  for ( i = &v8[result]; i != v8; *(_QWORD *)result |= 1LL << v11 )
  {
    v11 = *v8++;
    result = *a3 + ((v11 >> 3) & 0x1FF8);
  }
  return result;
}
