// Function: sub_880320
// Address: 0x880320
//
__int64 *__fastcall sub_880320(__int64 *a1, char a2, __int64 a3, char a4, __int64 *a5)
{
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r12
  char v11; // al
  __int64 *v12; // rdx
  __int64 *result; // rax

  v8 = qword_4F60010;
  if ( qword_4F60010 )
    qword_4F60010 = *(_QWORD *)qword_4F60010;
  else
    v8 = sub_823970(40);
  *(_BYTE *)(v8 + 16) = a2;
  *(_QWORD *)(v8 + 32) = a3;
  *(_BYTE *)(v8 + 24) = a4;
  v9 = *a5;
  *(_QWORD *)v8 = 0;
  *(_QWORD *)(v8 + 8) = v9;
  v10 = *a1;
  v11 = *(_BYTE *)(*a1 + 80);
  if ( (unsigned __int8)(v11 - 4) <= 1u || v11 == 3 && (unsigned int)sub_8D3A70(*(_QWORD *)(v10 + 88)) )
    v12 = (__int64 *)(*(_QWORD *)(v10 + 96) + 128LL);
  else
    v12 = *(__int64 **)(v10 + 96);
  result = (__int64 *)*v12;
  if ( *v12 )
  {
    do
    {
      v12 = result;
      result = (__int64 *)*result;
    }
    while ( result );
  }
  *v12 = v8;
  return result;
}
