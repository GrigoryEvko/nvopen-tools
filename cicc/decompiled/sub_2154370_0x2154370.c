// Function: sub_2154370
// Address: 0x2154370
//
unsigned __int8 __fastcall sub_2154370(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, const char *a5)
{
  __int64 v7; // rsi
  __int64 v8; // rax
  unsigned __int8 result; // al
  signed int v10; // edi
  const char *v11; // rax
  __int64 v12; // r13
  unsigned int v13; // eax

  v7 = *(_QWORD *)(a2 + 32) + 40LL * a3;
  switch ( *(_BYTE *)v7 )
  {
    case 0:
      v10 = *(_DWORD *)(v7 + 8);
      if ( v10 <= 0 )
      {
        result = (unsigned __int8)sub_214F100(a1, v10, a4);
      }
      else if ( v10 == 1 )
      {
        v12 = sub_1263B40(a4, "__local_depot");
        v13 = sub_396DD70(a1);
        result = sub_16E7A90(v12, v13);
      }
      else
      {
        v11 = (const char *)sub_2189340();
        result = sub_1263B40(a4, v11);
      }
      break;
    case 1:
      if ( a5 )
        result = sub_214F160(a1, v7, a5, a4);
      else
        result = sub_16E7AB0(a4, *(_QWORD *)(v7 + 24));
      break;
    case 2:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 0xA:
      v8 = sub_396EAF0(a1, *(_QWORD *)(v7 + 24));
      goto LABEL_3;
    case 3:
      result = sub_21523A0(a1, *(__int64 **)(v7 + 24), a4);
      break;
    case 4:
      v8 = sub_1DD5A70(*(_QWORD *)(v7 + 24));
LABEL_3:
      result = sub_38E2490(v8, a4, *(_QWORD *)(a1 + 240));
      break;
  }
  return result;
}
