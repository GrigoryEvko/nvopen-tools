// Function: sub_2D9F630
// Address: 0x2d9f630
//
__int64 __fastcall sub_2D9F630(__int64 *a1, __int64 a2, int *a3, __int64 *a4, __int64 *a5)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  int v11; // esi
  __int64 result; // rax

  *a1 = 0;
  v8 = sub_22077B0(0x80u);
  if ( v8 )
  {
    v9 = *a5;
    v10 = *a4;
    *(_QWORD *)v8 = off_49D41B0;
    v11 = *a3;
    *(_QWORD *)(v8 + 32) = v9;
    *(_DWORD *)(v8 + 20) = v11;
    *(_QWORD *)(v8 + 8) = 0x100000001LL;
    *(_BYTE *)(v8 + 16) = 1;
    *(_QWORD *)(v8 + 24) = v10;
    *(_BYTE *)(v8 + 48) = 0;
    *(_DWORD *)(v8 + 52) = 0;
    *(_QWORD *)(v8 + 56) = v8 + 72;
    *(_QWORD *)(v8 + 64) = 0x600000000LL;
    *(_QWORD *)(v8 + 120) = 0;
  }
  a1[1] = v8;
  result = v8 + 16;
  *a1 = result;
  return result;
}
