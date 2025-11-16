// Function: sub_1CCB280
// Address: 0x1ccb280
//
bool __fastcall sub_1CCB280(__int64 a1)
{
  bool result; // al

  result = 0;
  if ( !(*(_DWORD *)(*(_QWORD *)a1 + 8LL) >> 8) )
    return *(_DWORD *)(**(_QWORD **)(a1 - 24) + 8LL) >> 8 == 3;
  return result;
}
