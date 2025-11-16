// Function: sub_1319280
// Address: 0x1319280
//
int __fastcall sub_1319280(__int64 a1, __int64 a2)
{
  unsigned int v2; // ebx
  __int64 v3; // rdx

  if ( dword_4F96B60 )
  {
    v2 = 0;
    do
    {
      v3 = v2++;
      sub_131C7F0(a1, a2 + 224 * v3 + 78984);
    }
    while ( dword_4F96B60 > v2 );
  }
  sub_130B050(a1, a2 + 10536);
  sub_131C560(a1, *(_QWORD *)(a2 + 78936));
  sub_134ACC0(a1, a2 + 10648);
  return sub_130B050(a1, a2 + 10408);
}
