// Function: sub_2A191C0
// Address: 0x2a191c0
//
__int64 __fastcall sub_2A191C0(_BYTE *a1, _BYTE *a2, unsigned int a3, unsigned int a4, unsigned __int8 a5, __int64 a6)
{
  unsigned int v8; // r13d
  unsigned int v9; // ebx
  unsigned __int64 v10; // rbx
  void (__fastcall *v11)(unsigned __int64); // rax
  unsigned int v13; // r15d
  unsigned int v14; // r12d
  int v15; // eax
  int v16; // eax
  int v17; // [rsp+8h] [rbp-48h]
  unsigned __int64 v19[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( a1 == a2 )
    return 1;
  if ( *a1 == 61 )
  {
    v8 = 1;
    if ( *a2 == 61 )
      return v8;
  }
  sub_2297CA0((__int64 *)v19, a6, (__int64)a1, a2);
  if ( !v19[0] )
    return 1;
  v8 = (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v19[0] + 24LL))(v19[0]);
  if ( (_BYTE)v8 )
  {
    v10 = v19[0];
    v8 = 0;
  }
  else
  {
    v9 = 1;
    if ( a3 > 1 )
    {
      while ( ((*(__int64 (__fastcall **)(unsigned __int64, _QWORD))(*(_QWORD *)v19[0] + 48LL))(v19[0], v9) & 2) != 0 )
      {
        if ( a3 == ++v9 )
          goto LABEL_17;
      }
    }
    else
    {
LABEL_17:
      v17 = (*(__int64 (__fastcall **)(unsigned __int64, _QWORD))(*(_QWORD *)v19[0] + 48LL))(v19[0], a3);
      if ( v17 != 2 )
      {
        v10 = v19[0];
        if ( (v17 & 1) == 0 || (v13 = a3 + 1, a4 < a3 + 1) )
        {
          if ( (v17 & 4) != 0 )
          {
            v13 = a3 + 1;
            if ( a4 >= a3 + 1 )
              goto LABEL_31;
            v8 = a5;
          }
          else
          {
LABEL_20:
            v8 = 1;
          }
          goto LABEL_12;
        }
        v14 = a3 + 1;
        do
        {
          v15 = (*(__int64 (__fastcall **)(unsigned __int64, _QWORD))(*(_QWORD *)v10 + 48LL))(v10, v14);
          if ( v15 == 1 )
            break;
          if ( (v15 & 4) != 0 )
          {
LABEL_35:
            v10 = v19[0];
            goto LABEL_12;
          }
          ++v14;
        }
        while ( a4 >= v14 );
        v10 = v19[0];
        if ( (v17 & 4) == 0 )
          goto LABEL_20;
LABEL_31:
        while ( 1 )
        {
          v16 = (*(__int64 (__fastcall **)(unsigned __int64, _QWORD))(*(_QWORD *)v10 + 48LL))(v10, v13);
          if ( v16 == 4 )
            break;
          if ( (v16 & 1) != 0 )
            goto LABEL_35;
          if ( a4 < ++v13 )
          {
            v10 = v19[0];
            v8 = a5;
            goto LABEL_12;
          }
        }
      }
    }
    v10 = v19[0];
    v8 = 1;
  }
LABEL_12:
  if ( v10 )
  {
    v11 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v10 + 8LL);
    if ( v11 == sub_228A6E0 )
      j_j___libc_free_0(v10);
    else
      v11(v10);
  }
  return v8;
}
