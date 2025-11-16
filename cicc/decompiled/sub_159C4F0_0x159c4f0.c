// Function: sub_159C4F0
// Address: 0x159c4f0
//
__int64 __fastcall sub_159C4F0(__int64 *a1)
{
  __int64 v1; // rbx
  __int64 result; // rax
  __int64 v3; // rax

  v1 = *a1;
  result = *(_QWORD *)(*a1 + 1840);
  if ( !result )
  {
    v3 = sub_1643320(a1);
    result = sub_159C470(v3, 1, 0);
    *(_QWORD *)(v1 + 1840) = result;
  }
  return result;
}
