// Function: sub_B445D0
// Address: 0xb445d0
//
__int64 __fastcall sub_B445D0(__int64 a1, char *a2)
{
  char v2; // al
  __int64 v3; // rbx
  __int64 v4; // rax
  __int16 v5; // dx
  char v6; // si
  char v7; // cl
  __int64 v8; // rdx

  v2 = *a2;
  if ( *a2 == 84 )
  {
    v3 = *((_QWORD *)a2 + 5);
    v4 = sub_AA5190(v3);
    if ( v4 )
    {
LABEL_3:
      v6 = v5;
      v7 = HIBYTE(v5);
LABEL_4:
      LOBYTE(v8) = v6;
      BYTE1(v8) = v7;
      v8 = (unsigned __int16)v8;
      goto LABEL_5;
    }
LABEL_9:
    v7 = 0;
    v6 = 0;
    goto LABEL_4;
  }
  if ( v2 == 34 )
  {
    v3 = *((_QWORD *)a2 - 12);
    v4 = sub_AA5190(v3);
    if ( v4 )
      goto LABEL_3;
    goto LABEL_9;
  }
  if ( v2 == 40 )
    goto LABEL_10;
  v3 = *((_QWORD *)a2 + 5);
  v4 = *((_QWORD *)a2 + 4);
  v8 = 0;
  v6 = 1;
LABEL_5:
  if ( v4 != v3 + 48 )
  {
    LOBYTE(v8) = v6;
    *(_QWORD *)a1 = v4;
    *(_QWORD *)(a1 + 8) = v8;
    *(_BYTE *)(a1 + 16) = 1;
    return a1;
  }
LABEL_10:
  *(_BYTE *)(a1 + 16) = 0;
  return a1;
}
