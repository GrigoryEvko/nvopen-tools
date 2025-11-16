// Function: sub_737880
// Address: 0x737880
//
__int64 __fastcall sub_737880(__int64 a1)
{
  __int64 v2; // rdi
  _QWORD **v3; // rbx
  _QWORD *v4; // rax
  __int64 v6; // rdx

  v2 = qword_4F07A10;
  if ( !qword_4F07A10 )
  {
    qword_4F07A10 = sub_881A70(0xFFFFFFFFLL, 512, 31, 51);
    v2 = qword_4F07A10;
  }
  v3 = (_QWORD **)sub_881B20(v2, a1, 1);
  v4 = *v3;
  if ( !*v3 )
  {
    v4 = (_QWORD *)sub_822B10(16);
    v6 = qword_4F07A18;
    *v4 = a1;
    v4[1] = v6;
    qword_4F07A18 = v6 + 1;
    *v3 = v4;
  }
  return v4[1];
}
