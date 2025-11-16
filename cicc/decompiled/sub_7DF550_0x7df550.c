// Function: sub_7DF550
// Address: 0x7df550
//
__int64 sub_7DF550()
{
  __int64 result; // rax
  int v1; // [rsp+4h] [rbp-Ch] BYREF
  __int64 v2; // [rsp+8h] [rbp-8h] BYREF

  sub_622920(unk_4F06871, &v2, &v1);
  result = -1;
  if ( v2 * (unsigned __int64)dword_4F06BA0 <= 0x3F )
    result = (1LL << ((unsigned __int8)v2 * (unsigned __int8)dword_4F06BA0)) - 1;
  qword_4F18920 = 0;
  qword_4F18918 = 0;
  unk_4D03EC0 = result;
  qword_4F18900 = 0;
  qword_4F188F8 = 0;
  qword_4F18908 = 0;
  qword_4F18930 = 0;
  qword_4F18940 = 0;
  return result;
}
