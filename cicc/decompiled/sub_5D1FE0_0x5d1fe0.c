// Function: sub_5D1FE0
// Address: 0x5d1fe0
//
__int64 __fastcall sub_5D1FE0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v4; // r15
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // r14
  unsigned __int64 v10; // rax
  __int64 v11; // r13
  unsigned __int64 v12; // rax
  __int64 v13; // [rsp+8h] [rbp-48h]
  _DWORD v14[13]; // [rsp+1Ch] [rbp-34h] BYREF

  if ( unk_4D045E8 <= 0x59u )
    sub_684AA0(7, 3687, a1 + 56);
  v2 = *(_QWORD *)(a1 + 32);
  if ( *(_BYTE *)(v2 + 10) )
  {
    v4 = *(_QWORD **)v2;
    v5 = *(_QWORD *)(v2 + 40);
    v6 = 0;
    if ( *(_QWORD *)v2 )
    {
      v6 = v4[5];
      v4 = (_QWORD *)*v4;
      if ( v4 )
        v4 = (_QWORD *)v4[5];
    }
    if ( !(unsigned int)sub_5D19D0(a1) )
    {
      v7 = *(_QWORD *)(a2 + 328);
      if ( !v7 )
      {
        v7 = sub_725F60();
        *(_QWORD *)(a2 + 328) = v7;
      }
      if ( (*(_BYTE *)(v7 + 52) & 2) != 0 )
      {
        sub_684AA0(7, 3791, a1 + 56);
        v7 = *(_QWORD *)(a2 + 328);
      }
      *(_BYTE *)(v7 + 52) |= 1u;
      v13 = *(_QWORD *)(a2 + 328);
      if ( v5 )
      {
        if ( (int)sub_6210B0(v5, 0) <= 0 )
        {
          sub_6849F0(7, 3685, a1 + 56, "__cluster_dims__");
        }
        else
        {
          v8 = sub_620FD0(v5, v14);
          if ( v14[0] || v8 > 0x7FFFFFFF )
            sub_684AA0(7, 3686, a1 + 56);
          else
            *(_DWORD *)(v13 + 20) = v8;
        }
      }
      else
      {
        *(_DWORD *)(v13 + 20) = 1;
      }
      v9 = *(_QWORD *)(a2 + 328);
      if ( v6 )
      {
        if ( (int)sub_6210B0(v6, 0) <= 0 )
        {
          sub_6849F0(7, 3685, a1 + 56, "__cluster_dims__");
        }
        else
        {
          v10 = sub_620FD0(v6, v14);
          if ( v14[0] || v10 > 0x7FFFFFFF )
            sub_684AA0(7, 3686, a1 + 56);
          else
            *(_DWORD *)(v9 + 24) = v10;
        }
      }
      else
      {
        *(_DWORD *)(v9 + 24) = 1;
      }
      v11 = *(_QWORD *)(a2 + 328);
      if ( v4 )
      {
        if ( (int)sub_6210B0(v4, 0) <= 0 )
        {
          sub_6849F0(7, 3685, a1 + 56, "__cluster_dims__");
        }
        else
        {
          v12 = sub_620FD0(v4, v14);
          if ( v14[0] || v12 > 0x7FFFFFFF )
            sub_684AA0(7, 3686, a1 + 56);
          else
            *(_DWORD *)(v11 + 28) = v12;
        }
      }
      else
      {
        *(_DWORD *)(v11 + 28) = 1;
      }
    }
  }
  else
  {
    *(_BYTE *)(a2 + 199) |= 0x20u;
  }
  return a2;
}
