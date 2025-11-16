// Function: sub_88F9D0
// Address: 0x88f9d0
//
void __fastcall sub_88F9D0(__int64 *a1, int a2)
{
  __int64 *v2; // r14
  int v3; // r13d
  __int64 *v4; // rbx
  char v5; // al

  if ( a1 )
  {
    v2 = 0;
    v3 = 0;
    v4 = a1;
    while ( 1 )
    {
      v5 = *((_BYTE *)v4 + 56);
      if ( (v5 & 1) != 0 )
      {
        if ( a2 )
        {
          v2 = v4;
          v3 = 1;
          sub_6851C0(0x349u, (_DWORD *)(v4[1] + 48));
LABEL_5:
          v4 = (__int64 *)*v4;
          if ( !v4 )
            return;
        }
        else
        {
          v2 = v4;
          v3 = 1;
          if ( (v5 & 0x10) == 0 )
            goto LABEL_5;
LABEL_9:
          if ( (v5 & 0x60) != 0 )
            goto LABEL_13;
          if ( !*v4 )
            return;
          if ( *((_DWORD *)v4 + 15) != *(_DWORD *)(*v4 + 60) )
          {
            sub_6851C0(0x778u, (_DWORD *)(v4[1] + 48));
            goto LABEL_5;
          }
          v4 = (__int64 *)*v4;
        }
      }
      else
      {
        if ( !a2 && (v5 & 0x10) != 0 )
          goto LABEL_9;
LABEL_13:
        if ( (v4[7] & 1) != 0 || !v3 )
          goto LABEL_5;
        sub_684AA0(8u, 0x132u, (_DWORD *)(v2[1] + 48));
        v4 = (__int64 *)*v4;
        if ( !v4 )
          return;
      }
    }
  }
}
