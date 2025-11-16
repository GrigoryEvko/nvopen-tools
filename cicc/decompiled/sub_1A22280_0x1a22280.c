// Function: sub_1A22280
// Address: 0x1a22280
//
__int64 __fastcall sub_1A22280(__int64 a1)
{
  __int64 result; // rax
  void *v3; // rsi
  void *v4; // rdi
  __int64 v5; // rdx

  result = 0;
  v3 = *(void **)(a1 + 504);
  v4 = *(void **)(a1 + 496);
  if ( v4 != v3 )
  {
    sub_1B3B3D0(v4);
    v5 = *(_QWORD *)(a1 + 496);
    result = 1;
    if ( v5 != *(_QWORD *)(a1 + 504) )
      *(_QWORD *)(a1 + 504) = v5;
  }
  return result;
}
