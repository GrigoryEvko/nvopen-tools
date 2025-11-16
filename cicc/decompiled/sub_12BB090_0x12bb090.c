// Function: sub_12BB090
// Address: 0x12bb090
//
__int64 __fastcall sub_12BB090(__int64 *a1)
{
  char v1; // r12
  __int64 v2; // r13
  __int64 v4; // rax
  unsigned int v5; // r14d

  v1 = byte_4F92D70;
  if ( byte_4F92D70 || !dword_4C6F008 )
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v2 = qword_4F92D80;
    sub_16C30C0(qword_4F92D80);
    if ( !a1 )
    {
      v5 = 5;
      goto LABEL_16;
    }
    v1 = 1;
  }
  else
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v2 = qword_4F92D80;
    if ( !a1 )
      return 5;
  }
  v4 = sub_22077B0(224);
  if ( v4 )
  {
    *(_QWORD *)v4 = 0;
    v5 = 0;
    *(_QWORD *)(v4 + 48) = v4 + 64;
    *(_QWORD *)(v4 + 8) = 0;
    *(_QWORD *)(v4 + 16) = 0;
    *(_QWORD *)(v4 + 24) = 0;
    *(_QWORD *)(v4 + 32) = 0;
    *(_QWORD *)(v4 + 40) = 0;
    *(_QWORD *)(v4 + 56) = 0;
    *(_BYTE *)(v4 + 64) = 0;
    *(_QWORD *)(v4 + 80) = v4 + 96;
    *(_QWORD *)(v4 + 88) = 0;
    *(_BYTE *)(v4 + 96) = 0;
    *(_QWORD *)(v4 + 184) = 0;
    *(_QWORD *)(v4 + 192) = 0;
    *(_QWORD *)(v4 + 200) = 0;
    *(_QWORD *)(v4 + 112) = 0;
    *(_QWORD *)(v4 + 120) = 0;
    *(_QWORD *)(v4 + 128) = 0;
    *(_QWORD *)(v4 + 136) = 0;
    *(_QWORD *)(v4 + 144) = 0;
    *(_QWORD *)(v4 + 152) = 0;
    *(_QWORD *)(v4 + 160) = 0;
    *(_QWORD *)(v4 + 168) = 0;
    *(_DWORD *)(v4 + 176) = 0;
    *(_QWORD *)(v4 + 208) = 0;
    *(_QWORD *)(v4 + 216) = 0;
    *a1 = v4;
  }
  else
  {
    v5 = 1;
  }
  if ( !v1 )
    return v5;
LABEL_16:
  sub_16C30E0(v2);
  return v5;
}
