// Function: sub_30E2940
// Address: 0x30e2940
//
__int64 *__fastcall sub_30E2940(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax

  if ( dword_5031048 == 2 )
  {
    v4 = sub_22077B0(0x108u);
    if ( v4 )
    {
      *(_QWORD *)(v4 + 248) = a2;
      *(_QWORD *)v4 = off_49D8B58;
      *(_QWORD *)(v4 + 168) = sub_30E0F20;
      *(_QWORD *)(v4 + 8) = v4 + 24;
      *(_QWORD *)(v4 + 16) = 0x1000000000LL;
      *(_QWORD *)(v4 + 184) = 0;
      *(_QWORD *)(v4 + 192) = 0;
      *(_QWORD *)(v4 + 200) = 0;
      *(_DWORD *)(v4 + 208) = 0;
      *(_QWORD *)(v4 + 216) = 0;
      *(_QWORD *)(v4 + 224) = 0;
      *(_QWORD *)(v4 + 232) = 0;
      *(_DWORD *)(v4 + 240) = 0;
      *(_QWORD *)(v4 + 256) = a3;
      *(_QWORD *)(v4 + 152) = v4;
      *(_QWORD *)(v4 + 176) = sub_30E1780;
    }
    goto LABEL_7;
  }
  if ( dword_5031048 > 2 )
  {
    if ( dword_5031048 == 3 )
    {
      v4 = sub_22077B0(0x108u);
      if ( v4 )
      {
        *(_QWORD *)(v4 + 248) = a2;
        *(_QWORD *)v4 = off_49D8B98;
        *(_QWORD *)(v4 + 8) = v4 + 24;
        *(_QWORD *)(v4 + 16) = 0x1000000000LL;
        *(_QWORD *)(v4 + 184) = 0;
        *(_QWORD *)(v4 + 192) = 0;
        *(_QWORD *)(v4 + 200) = 0;
        *(_DWORD *)(v4 + 208) = 0;
        *(_QWORD *)(v4 + 216) = 0;
        *(_QWORD *)(v4 + 224) = 0;
        *(_QWORD *)(v4 + 232) = 0;
        *(_DWORD *)(v4 + 240) = 0;
        *(_QWORD *)(v4 + 256) = a3;
        *(_QWORD *)(v4 + 152) = v4;
        *(_QWORD *)(v4 + 168) = sub_30E0F50;
        *(_QWORD *)(v4 + 176) = sub_30E1560;
      }
      goto LABEL_7;
    }
  }
  else
  {
    if ( !dword_5031048 )
    {
      v4 = sub_22077B0(0x108u);
      if ( v4 )
      {
        *(_QWORD *)(v4 + 184) = 0;
        *(_QWORD *)v4 = off_49D8AD8;
        *(_QWORD *)(v4 + 16) = 0x1000000000LL;
        *(_QWORD *)(v4 + 168) = sub_30E0EC0;
        *(_QWORD *)(v4 + 8) = v4 + 24;
        *(_QWORD *)(v4 + 192) = 0;
        *(_QWORD *)(v4 + 200) = 0;
        *(_DWORD *)(v4 + 208) = 0;
        *(_QWORD *)(v4 + 216) = 0;
        *(_QWORD *)(v4 + 224) = 0;
        *(_QWORD *)(v4 + 232) = 0;
        *(_DWORD *)(v4 + 240) = 0;
        *(_QWORD *)(v4 + 248) = a2;
        *(_QWORD *)(v4 + 256) = a3;
        *(_QWORD *)(v4 + 152) = v4;
        *(_QWORD *)(v4 + 176) = sub_30E1450;
      }
      goto LABEL_7;
    }
    if ( dword_5031048 == 1 )
    {
      v4 = sub_22077B0(0x108u);
      if ( v4 )
      {
        *(_QWORD *)(v4 + 248) = a2;
        *(_QWORD *)v4 = off_49D8B18;
        *(_QWORD *)(v4 + 168) = sub_30E0EF0;
        *(_QWORD *)(v4 + 8) = v4 + 24;
        *(_QWORD *)(v4 + 16) = 0x1000000000LL;
        *(_QWORD *)(v4 + 184) = 0;
        *(_QWORD *)(v4 + 192) = 0;
        *(_QWORD *)(v4 + 200) = 0;
        *(_DWORD *)(v4 + 208) = 0;
        *(_QWORD *)(v4 + 216) = 0;
        *(_QWORD *)(v4 + 224) = 0;
        *(_QWORD *)(v4 + 232) = 0;
        *(_DWORD *)(v4 + 240) = 0;
        *(_QWORD *)(v4 + 256) = a3;
        *(_QWORD *)(v4 + 152) = v4;
        *(_QWORD *)(v4 + 176) = sub_30E1670;
      }
LABEL_7:
      *a1 = v4;
      return a1;
    }
  }
  *a1 = 0;
  return a1;
}
