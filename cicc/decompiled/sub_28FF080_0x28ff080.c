// Function: sub_28FF080
// Address: 0x28ff080
//
__int64 __fastcall sub_28FF080(__int64 a1, __int64 a2, unsigned __int8 a3, char a4)
{
  __int64 *v4; // rbx
  __int64 result; // rax
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // rdi
  __int64 v13; // [rsp+18h] [rbp-38h]

  v4 = *(__int64 **)a1;
  result = *(unsigned int *)(a1 + 8);
  v6 = *(_QWORD *)a1 + 8 * result;
  if ( *(_QWORD *)a1 != v6 )
  {
    do
    {
      v9 = *v4;
      v10 = *(_QWORD *)(*v4 + 72);
      if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
        v7 = sub_AC9350((__int64 **)v10);
      else
        v7 = sub_AC9EC0((__int64 **)v10);
      v8 = a3;
      BYTE1(v8) = a4;
      v13 = v8;
      result = (__int64)sub_BD2C40(80, unk_3F10A10);
      if ( result )
        result = sub_B4D460(result, v7, v9, a2, v13);
      ++v4;
    }
    while ( (__int64 *)v6 != v4 );
  }
  return result;
}
