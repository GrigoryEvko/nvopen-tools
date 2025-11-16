// Function: sub_E214D0
// Address: 0xe214d0
//
unsigned __int64 __fastcall sub_E214D0(__int64 a1, __int64 a2, char a3)
{
  _QWORD *v4; // rdx
  unsigned __int64 result; // rax
  unsigned __int64 *v6; // rax
  unsigned __int64 *v7; // r12
  unsigned __int64 v8; // rdx

  v4 = *(_QWORD **)(a1 + 16);
  result = (*v4 + v4[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v4[1] = result - *v4 + 40;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v6 = (unsigned __int64 *)sub_22077B0(32);
    v7 = v6;
    if ( v6 )
    {
      *v6 = 0;
      v6[1] = 0;
      v6[2] = 0;
      v6[3] = 0;
    }
    result = sub_2207820(4096);
    v8 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 16) = v7;
    *v7 = result;
    v7[3] = v8;
    v7[2] = 4096;
    v7[1] = 40;
  }
  if ( !result )
  {
    MEMORY[0x20] = 0;
    BUG();
  }
  *(_BYTE *)(result + 32) = 0;
  *(_DWORD *)(result + 8) = 11;
  *(_QWORD *)(result + 16) = 0;
  *(_QWORD *)result = &unk_49E1028;
  *(_QWORD *)(result + 24) = 0;
  *(_BYTE *)(result + 32) = a3;
  return result;
}
