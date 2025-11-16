// Function: sub_2E79040
// Address: 0x2e79040
//
unsigned __int64 __fastcall sub_2E79040(__int64 a1)
{
  __int64 v1; // rax
  int v2; // ecx
  int v3; // edx
  __int64 *i; // rax
  __int64 v5; // rdx
  unsigned __int64 result; // rax

  *(_BYTE *)(*(_QWORD *)(a1 + 328) + 260LL) = 1;
  v1 = *(_QWORD *)(a1 + 328);
  v2 = *(_DWORD *)(v1 + 252);
  v3 = *(_DWORD *)(v1 + 256);
  for ( i = *(__int64 **)(v1 + 8); (__int64 *)(a1 + 320) != i; i = (__int64 *)i[1] )
  {
    if ( *((_DWORD *)i + 64) != v3 || *((_DWORD *)i + 63) != v2 )
    {
      v5 = *i;
      *((_BYTE *)i + 260) = 1;
      *(_BYTE *)((v5 & 0xFFFFFFFFFFFFFFF8LL) + 261) = 1;
      v2 = *((_DWORD *)i + 63);
      v3 = *((_DWORD *)i + 64);
    }
  }
  result = *(_QWORD *)(a1 + 320) & 0xFFFFFFFFFFFFFFF8LL;
  *(_BYTE *)(result + 261) = 1;
  return result;
}
