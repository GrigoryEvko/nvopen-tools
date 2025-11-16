// Function: sub_728C90
// Address: 0x728c90
//
__int64 __fastcall sub_728C90(__int64 a1, __int64 a2)
{
  const char *v2; // rdx

  v2 = *(const char **)(a1 + 8);
  if ( !strcmp(v2, "_M_function_name") )
  {
    if ( !*((_QWORD *)&xmmword_4F07A80 + 1) && *(_QWORD *)(a1 + 120) == a2 )
    {
      *((_QWORD *)&xmmword_4F07A80 + 1) = a1;
      return 1;
    }
    return 0;
  }
  if ( !strcmp(v2, "_M_file_name") )
  {
    if ( !(_QWORD)xmmword_4F07A80 && a2 == *(_QWORD *)(a1 + 120) )
    {
      *(_QWORD *)&xmmword_4F07A80 = a1;
      return 1;
    }
    return 0;
  }
  if ( strcmp(v2, "_M_column") )
  {
    if ( !strcmp(v2, "_M_line") && !(_QWORD)xmmword_4F07A90 && (unsigned int)sub_8D2780(*(_QWORD *)(a1 + 120)) )
    {
      *(_QWORD *)&xmmword_4F07A90 = a1;
      return 1;
    }
    return 0;
  }
  if ( *((_QWORD *)&xmmword_4F07A90 + 1) || !(unsigned int)sub_8D2780(*(_QWORD *)(a1 + 120)) )
    return 0;
  *((_QWORD *)&xmmword_4F07A90 + 1) = a1;
  return 1;
}
