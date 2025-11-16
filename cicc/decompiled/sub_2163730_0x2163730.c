// Function: sub_2163730
// Address: 0x2163730
//
__int64 __fastcall sub_2163730(__int64 a1, __int16 ***a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  result = a1;
  v3 = a1 + 16;
  if ( a2 == &off_4A02620 )
  {
    *(_QWORD *)a1 = v3;
    strcpy((char *)(a1 + 16), ".f32");
    *(_QWORD *)(a1 + 8) = 4;
    return result;
  }
  if ( a2 == &off_4A02760 )
    goto LABEL_13;
  if ( a2 == &off_4A026A0 )
    goto LABEL_15;
  if ( a2 == &off_4A02520 )
  {
    *(_QWORD *)a1 = v3;
    strcpy((char *)(a1 + 16), ".f64");
    *(_QWORD *)(a1 + 8) = 4;
    return result;
  }
  if ( a2 == &off_4A02460 )
  {
    *(_QWORD *)a1 = v3;
    *(_DWORD *)(a1 + 16) = 842097198;
    *(_BYTE *)(a1 + 20) = 56;
    *(_QWORD *)(a1 + 8) = 5;
    *(_BYTE *)(a1 + 21) = 0;
    return result;
  }
  if ( a2 == &off_4A024A0 )
  {
    *(_QWORD *)a1 = v3;
    strcpy((char *)(a1 + 16), ".b64");
    *(_QWORD *)(a1 + 8) = 4;
    return result;
  }
  if ( a2 == &off_4A025A0 )
  {
LABEL_15:
    *(_QWORD *)a1 = v3;
    strcpy((char *)(a1 + 16), ".b32");
    *(_QWORD *)(a1 + 8) = 4;
    return result;
  }
  if ( a2 == &off_4A02720 )
  {
LABEL_13:
    *(_QWORD *)a1 = v3;
    strcpy((char *)(a1 + 16), ".b16");
    *(_QWORD *)(a1 + 8) = 4;
    return result;
  }
  *(_QWORD *)a1 = v3;
  if ( a2 == &off_4A027A0 )
  {
    *(_DWORD *)(a1 + 16) = 1701998638;
    *(_BYTE *)(a1 + 20) = 100;
    *(_QWORD *)(a1 + 8) = 5;
    *(_BYTE *)(a1 + 21) = 0;
  }
  else if ( a2 == (__int16 ***)&off_4A026E0 )
  {
    strcpy((char *)(a1 + 16), "!Special!");
    *(_QWORD *)(a1 + 8) = 9;
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 8;
    strcpy((char *)(a1 + 16), "INTERNAL");
  }
  return result;
}
