// Function: sub_1371D60
// Address: 0x1371d60
//
__int64 __fastcall sub_1371D60(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rdx
  _QWORD *v3; // rcx
  __int64 v4; // rax
  bool v5; // cf
  __int64 result; // rax
  __int16 v7; // cx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int16 v10; // dx
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // [rsp+18h] [rbp-28h] BYREF
  __int64 v14; // [rsp+20h] [rbp-20h] BYREF
  __int16 v15; // [rsp+28h] [rbp-18h]

  v2 = *(_QWORD **)(a2 + 128);
  v3 = &v2[*(unsigned int *)(a2 + 136)];
  if ( v3 == v2 )
  {
    v13 = -1;
LABEL_9:
    v9 = sub_1370BB0(&v13);
    v15 = v10;
    v14 = v9;
    v11 = sub_1371CE0(1, 0, (__int64)&v14);
    result = v12;
    v8 = v11;
    v7 = result;
    goto LABEL_7;
  }
  v4 = 0;
  do
  {
    v5 = __CFADD__(*v2, v4);
    v4 += *v2;
    if ( v5 )
      v4 = -1;
    ++v2;
  }
  while ( v2 != v3 );
  result = ~v4;
  v7 = 12;
  v8 = 1;
  v13 = result;
  if ( result )
    goto LABEL_9;
LABEL_7:
  *(_QWORD *)(a2 + 160) = v8;
  *(_WORD *)(a2 + 168) = v7;
  return result;
}
