// Function: sub_72B800
// Address: 0x72b800
//
__int64 __fastcall sub_72B800(__int64 a1)
{
  int *v1; // rax
  __int64 v2; // r8

  v1 = (int *)(unk_4F072B8 + 16LL * *(int *)(a1 + 160));
  v2 = *(_QWORD *)(unk_4F073B0 + 8LL * v1[2]);
  if ( v2 )
    return *(_QWORD *)v1;
  return v2;
}
