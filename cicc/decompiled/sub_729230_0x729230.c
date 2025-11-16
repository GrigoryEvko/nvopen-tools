// Function: sub_729230
// Address: 0x729230
//
void __fastcall sub_729230(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  void *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // [rsp+8h] [rbp-38h]

  v3 = a3;
  v4 = qword_4F07B08;
  if ( !qword_4F07B08 )
    goto LABEL_5;
  if ( *(_DWORD *)(qword_4F07B08 + 8) != a2 )
  {
    if ( *(_DWORD *)(qword_4F07B08 + 12) == -1 )
      *(_DWORD *)(qword_4F07B08 + 12) = a2 - 1;
LABEL_5:
    v4 = sub_727AD0();
    v5 = base;
    if ( unk_4F07310 >= (unsigned __int64)qword_4F07AF8 )
    {
      if ( qword_4F07AF8 )
      {
        v6 = 16 * qword_4F07AF8;
        v7 = 2 * qword_4F07AF8;
      }
      else
      {
        v7 = 1024;
        v6 = 0x2000;
      }
      v5 = (void *)sub_822C60(base, 8 * qword_4F07AF8, v6);
      base = v5;
      qword_4F07AF8 = v7;
      if ( unk_4F07308 )
        goto LABEL_7;
    }
    else if ( unk_4F07308 )
    {
LABEL_7:
      *(_QWORD *)qword_4F07B08 = v4;
LABEL_8:
      *((_QWORD *)v5 + unk_4F07310++) = v4;
      qword_4F07B08 = v4;
      goto LABEL_9;
    }
    unk_4F07308 = v4;
    goto LABEL_8;
  }
LABEL_9:
  *(_DWORD *)(v4 + 16) = v3;
  *(_DWORD *)(v4 + 8) = a2;
  *(_DWORD *)(v4 + 12) = -1;
  *(_QWORD *)(v4 + 24) = a1;
  LODWORD(qword_4F07B20) = a2;
  qword_4F07B28 = v3 - a2;
  qword_4F07B38 = a1;
  HIDWORD(qword_4F07B20) = -1;
  dword_4F07B30 = 0;
}
