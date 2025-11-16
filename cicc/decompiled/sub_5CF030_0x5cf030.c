// Function: sub_5CF030
// Address: 0x5cf030
//
void __fastcall sub_5CF030(__int64 *a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // r15
  _QWORD *v5; // r14
  __int64 v6; // r13
  __int64 v7; // rax
  char *v8; // rax
  __int64 (__fastcall *v9)(__int64, __int64, __int64); // r12
  char v10; // al
  _QWORD **v11; // rdi

  if ( a2 )
  {
    v3 = *a1;
    v5 = a2;
    do
    {
      v6 = (__int64)v5;
      v5 = (_QWORD *)*v5;
      v7 = *(unsigned __int8 *)(v6 + 8);
      *(_QWORD *)(v6 + 48) = a3;
      v8 = (char *)&unk_496EE40 + 24 * v7;
      v9 = (__int64 (__fastcall *)(__int64, __int64, __int64))*((_QWORD *)v8 + 2);
      if ( (unsigned int)sub_5CCB50(*((char **)v8 + 1), v6, v3, 6) && *(_BYTE *)(v6 + 8) && v9 )
        v3 = v9(v6, v3, 6);
      *(_QWORD *)(v6 + 48) = 0;
    }
    while ( v5 );
    v10 = *(_BYTE *)(v3 + 140);
    if ( v10 == 7 || v10 == 12 && *(_BYTE *)(v3 + 184) == 8 )
    {
      v11 = (_QWORD **)(v3 + 104);
      if ( *(_QWORD *)(v3 + 104) )
        v11 = sub_5CB9F0(v11);
      *v11 = a2;
      *a1 = v3;
    }
    else
    {
      *a1 = sub_5CEF40(v3, a2);
    }
  }
}
