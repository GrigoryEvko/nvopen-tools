// Function: sub_1DE4D70
// Address: 0x1de4d70
//
void __fastcall sub_1DE4D70(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // r14
  __int64 v4; // rbx

  if ( a2 - a1 <= 224 )
  {
    sub_1DE3CF0(a1, a2);
  }
  else
  {
    v2 = (a2 - a1) >> 5;
    v3 = a1 + 16 * v2;
    v4 = (16 * v2) >> 4;
    sub_1DE4D70(a1, v3);
    sub_1DE4D70(v3, a2);
    sub_1DE4C10(a1, v3, a2, v4, (a2 - v3) >> 4);
  }
}
