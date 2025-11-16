// Function: sub_2B7D810
// Address: 0x2b7d810
//
__int64 __fastcall sub_2B7D810(_QWORD *a1, __int64 a2, char *a3, unsigned __int64 a4)
{
  __int64 v6; // r13
  __int64 v7; // rcx
  int v8; // edx
  __int64 *v10; // rbx
  unsigned __int16 v11; // dx

  v6 = *(_QWORD *)(a2 + 96);
  v7 = *(_QWORD *)(v6 + 8);
  v8 = *(unsigned __int8 *)(v7 + 8);
  if ( (unsigned int)(v8 - 17) <= 1 )
    LOBYTE(v8) = *(_BYTE *)(**(_QWORD **)(v7 + 16) + 8LL);
  if ( (_BYTE)v8 == 12 )
  {
    v10 = (__int64 *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8));
    LOBYTE(v11) = v10 != sub_2B13460(*(__int64 **)a2, (__int64)v10, (__int64)a1);
    HIBYTE(v11) = 1;
    v6 = sub_2B2EA50(a1, v6, v11);
  }
  return sub_2B7BF50((__int64)a1, v6, a3, a4);
}
