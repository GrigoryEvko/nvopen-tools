// Function: sub_16A89F0
// Address: 0x16a89f0
//
void __fastcall sub_16A89F0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rsi
  __int64 v5; // rdi
  unsigned __int64 v6; // rdx

  v3 = *((unsigned int *)a1 + 2);
  v4 = *a2;
  v5 = *a1;
  v6 = (unsigned __int64)(v3 + 63) >> 6;
  if ( (_DWORD)v6 )
  {
    v2 = 0;
    do
    {
      *(_QWORD *)(v5 + 8 * v2) |= *(_QWORD *)(v4 + 8 * v2);
      ++v2;
    }
    while ( (unsigned int)v6 != v2 );
  }
}
