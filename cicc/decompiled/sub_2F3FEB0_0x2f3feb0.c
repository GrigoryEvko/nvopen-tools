// Function: sub_2F3FEB0
// Address: 0x2f3feb0
//
__int64 sub_2F3FEB0()
{
  __int64 result; // rax

  if ( dword_5023688 == 1 )
  {
    result = sub_3595E80();
    if ( !result )
    {
      result = sub_22077B0(0xC0u);
      if ( result )
      {
        *(_QWORD *)(result + 8) = 0;
        *(_DWORD *)(result + 24) = 4;
        *(_QWORD *)(result + 16) = &unk_502343C;
        *(_QWORD *)(result + 56) = result + 104;
        *(_QWORD *)(result + 32) = 0;
        *(_QWORD *)(result + 40) = 0;
        *(_QWORD *)(result + 48) = 0;
        *(_QWORD *)(result + 64) = 1;
        *(_QWORD *)(result + 72) = 0;
        *(_QWORD *)(result + 80) = 0;
        *(_QWORD *)(result + 96) = 0;
        *(_QWORD *)(result + 104) = 0;
        *(_QWORD *)(result + 112) = result + 160;
        *(_QWORD *)(result + 120) = 1;
        *(_QWORD *)(result + 128) = 0;
        *(_QWORD *)(result + 136) = 0;
        *(_QWORD *)(result + 152) = 0;
        *(_QWORD *)(result + 160) = 0;
        *(_BYTE *)(result + 168) = 0;
        *(_QWORD *)(result + 176) = 0;
        *(_DWORD *)(result + 184) = 0;
        *(_QWORD *)result = off_4A2AD60;
        *(_BYTE *)(result + 188) = 1;
        *(_DWORD *)(result + 88) = 1065353216;
        *(_DWORD *)(result + 144) = 1065353216;
      }
    }
  }
  else if ( dword_5023688 == 2 )
  {
    result = sub_22077B0(0xC0u);
    if ( result )
    {
      *(_QWORD *)(result + 8) = 0;
      *(_DWORD *)(result + 24) = 4;
      *(_QWORD *)(result + 16) = &unk_502343C;
      *(_QWORD *)(result + 56) = result + 104;
      *(_QWORD *)(result + 32) = 0;
      *(_QWORD *)(result + 40) = 0;
      *(_QWORD *)(result + 48) = 0;
      *(_QWORD *)(result + 64) = 1;
      *(_QWORD *)(result + 72) = 0;
      *(_QWORD *)(result + 80) = 0;
      *(_QWORD *)(result + 96) = 0;
      *(_QWORD *)(result + 104) = 0;
      *(_QWORD *)(result + 112) = result + 160;
      *(_QWORD *)(result + 120) = 1;
      *(_QWORD *)(result + 128) = 0;
      *(_QWORD *)(result + 136) = 0;
      *(_QWORD *)(result + 152) = 0;
      *(_QWORD *)(result + 160) = 0;
      *(_BYTE *)(result + 168) = 0;
      *(_QWORD *)(result + 176) = 0;
      *(_DWORD *)(result + 184) = 0;
      *(_QWORD *)result = off_4A2AD60;
      *(_BYTE *)(result + 188) = 1;
      *(_DWORD *)(result + 88) = 1065353216;
      *(_DWORD *)(result + 144) = 1065353216;
    }
  }
  else
  {
    if ( dword_5023688 )
      BUG();
    result = sub_22077B0(0xC0u);
    if ( result )
    {
      *(_QWORD *)(result + 8) = 0;
      *(_DWORD *)(result + 24) = 4;
      *(_QWORD *)(result + 16) = &unk_502343C;
      *(_QWORD *)(result + 56) = result + 104;
      *(_QWORD *)(result + 32) = 0;
      *(_QWORD *)(result + 40) = 0;
      *(_QWORD *)(result + 48) = 0;
      *(_QWORD *)(result + 64) = 1;
      *(_QWORD *)(result + 72) = 0;
      *(_QWORD *)(result + 80) = 0;
      *(_QWORD *)(result + 96) = 0;
      *(_QWORD *)(result + 104) = 0;
      *(_QWORD *)(result + 112) = result + 160;
      *(_QWORD *)(result + 120) = 1;
      *(_QWORD *)(result + 128) = 0;
      *(_QWORD *)(result + 136) = 0;
      *(_QWORD *)(result + 152) = 0;
      *(_QWORD *)(result + 160) = 0;
      *(_BYTE *)(result + 168) = 0;
      *(_QWORD *)(result + 176) = 0;
      *(_DWORD *)(result + 184) = 0;
      *(_QWORD *)result = off_4A2AD60;
      *(_BYTE *)(result + 188) = 0;
      *(_DWORD *)(result + 88) = 1065353216;
      *(_DWORD *)(result + 144) = 1065353216;
    }
  }
  return result;
}
