// Function: sub_2E86530
// Address: 0x2e86530
//
unsigned __int64 __fastcall sub_2E86530(__int64 a1)
{
  __int64 v1; // rax
  int *v2; // rdx
  int v3; // eax
  unsigned __int64 v4; // r8

  v1 = *(_QWORD *)(a1 + 48);
  v2 = (int *)(v1 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v1 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 0;
  v3 = v1 & 7;
  v4 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 != 2 && (v4 = 0, v3 == 3) && *((_BYTE *)v2 + 5) )
    return *(_QWORD *)&v2[2 * *((unsigned __int8 *)v2 + 4) + 4 + 2 * (__int64)*v2];
  else
    return v4;
}
