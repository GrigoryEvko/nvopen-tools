// Function: sub_869840
// Address: 0x869840
//
void __fastcall sub_869840(__int64 a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r15
  _QWORD *v3; // r12
  _BYTE *v4; // r15
  _QWORD *v5; // rax
  int v6[13]; // [rsp+Ch] [rbp-34h] BYREF

  v1 = *(_QWORD **)(a1 + 256);
  if ( v1 )
  {
    v2 = 0;
    while ( (*(_BYTE *)(v1 - 1) & 1) == 0 )
    {
LABEL_3:
      v1 = (_QWORD *)*v1;
      if ( !v1 )
        return;
    }
    sub_7296C0(v6);
    v3 = sub_7271A0();
    sub_729730(v6[0]);
    if ( *(_QWORD *)(a1 + 264) )
      *v2 = v3;
    else
      *(_QWORD *)(a1 + 264) = v3;
    sub_7296F0(dword_4F04C58, v6);
    v4 = sub_727090();
    sub_729730(v6[0]);
    v4[16] = 55;
    *((_QWORD *)v4 + 3) = v3;
    v5 = (_QWORD *)v1[1];
    if ( v5 )
    {
      *v5 = v4;
      *((_QWORD *)v4 + 1) = v1[1];
      v1[1] = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 256) = v4;
    }
    v3[1] = v1;
    while ( 1 )
    {
      v3[2] = v1;
      v1 = (_QWORD *)*v1;
      if ( !v1 )
        break;
      if ( (*(_BYTE *)(v1 - 1) & 1) == 0 )
      {
        *(_QWORD *)v1[1] = 0;
        v1[1] = v4;
        *(_QWORD *)v4 = v1;
        v2 = v3;
        goto LABEL_3;
      }
    }
  }
}
