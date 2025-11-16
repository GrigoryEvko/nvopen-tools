// Function: sub_822F50
// Address: 0x822f50
//
__int64 __fastcall sub_822F50(int a1, __int64 a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // rax
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v8; // rdx

  v2 = 1;
  if ( a2 )
    v2 = a2;
  if ( (v2 & 7) != 0 )
    v2 += (int)(8 - (v2 & 7));
  v3 = *(_QWORD **)(unk_4F073B0 + 8LL * a1);
  v4 = v3[3];
  v5 = v3[2];
  if ( v2 + 8 > (unsigned __int64)(v4 - v5) )
  {
    v6 = *((unsigned __int8 *)v3 + 40);
    if ( !*((_BYTE *)v3 + 40) )
    {
      if ( (unsigned __int64)(v4 - v5) > 0x84F )
      {
        *(_QWORD *)(v5 + 32) = 0;
        *(_QWORD *)(v5 + 8) = v5 + 48;
        *(_QWORD *)(v5 + 16) = v5 + 48;
        v8 = qword_4F195E0;
        *(_QWORD *)(v5 + 24) = v4;
        *(_QWORD *)v5 = v8;
        *(_BYTE *)(v5 + 40) = 0;
        qword_4F195E0 = v5;
        v3[3] = v5;
      }
      *((_BYTE *)v3 + 40) = 1;
    }
    v3 = sub_822DE0(a1, v2 + 8, 0, (unsigned int)v6, v5, v6);
    v5 = v3[2];
  }
  v3[2] = v5 + v2;
  return v5;
}
