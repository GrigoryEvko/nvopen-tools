// Function: sub_39C2C50
// Address: 0x39c2c50
//
__int64 __fastcall sub_39C2C50(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rbx
  char *v5; // [rsp+0h] [rbp-30h] BYREF
  char v6; // [rsp+10h] [rbp-20h]
  char v7; // [rsp+11h] [rbp-1Fh]

  v2 = *(unsigned int *)(a1 + 8);
  v3 = *(_QWORD *)a1 + 32 * v2 - 32;
  if ( *(_QWORD *)(v3 + 16) == *(_DWORD *)(a1 + 152) )
  {
    *(_DWORD *)(a1 + 8) = v2 - 1;
    return 0;
  }
  else
  {
    v7 = 1;
    v5 = "debug_loc";
    v6 = 3;
    *(_QWORD *)(v3 + 8) = sub_396F530(a2, (__int64)&v5);
    return 1;
  }
}
