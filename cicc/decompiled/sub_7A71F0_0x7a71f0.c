// Function: sub_7A71F0
// Address: 0x7a71f0
//
void __fastcall sub_7A71F0(_QWORD *a1)
{
  __int64 v1; // rdx
  unsigned __int64 v2; // r8
  __int64 v3; // rsi
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rcx
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rtt

  v1 = a1[5];
  if ( v1 )
  {
    v2 = a1[1];
    v3 = unk_4F06AC0;
    v4 = a1[2];
    if ( v4 <= ~v1 )
    {
      v4 += v1;
      a1[2] = v4;
      v5 = dword_4F06BA0;
      if ( v4 < dword_4F06BA0 )
        goto LABEL_6;
    }
    else
    {
      v5 = dword_4F06BA0;
      if ( v4 < dword_4F06BA0 )
        goto LABEL_6;
    }
    v7 = v4;
    v6 = v4 / v5;
    a1[2] = v7 % v5;
    if ( v2 <= v3 - v6 )
      a1[1] = v2 + v6;
  }
LABEL_6:
  a1[4] = 0;
  a1[5] = 0;
}
