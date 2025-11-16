// Function: sub_70FEF0
// Address: 0x70fef0
//
__int64 __fastcall sub_70FEF0(__int64 a1, _BYTE *a2, _DWORD *a3, _DWORD *a4)
{
  __int64 result; // rax
  __int64 v8; // rdx

  for ( result = *(_QWORD *)(a1 + 128); *(_BYTE *)(result + 140) == 12; result = *(_QWORD *)(result + 160) )
    ;
  v8 = *(unsigned __int8 *)(result + 160);
  *a2 = v8;
  *a3 = byte_4B6DF90[v8];
  *a4 = *(_DWORD *)(result + 128) * dword_4F06BA0;
  return result;
}
