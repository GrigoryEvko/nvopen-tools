// Function: sub_1E80440
// Address: 0x1e80440
//
__int64 __fastcall sub_1E80440(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r15
  __int64 *v3; // r14
  __int64 v4; // r13
  __int64 v5; // r12
  _QWORD *v6; // rax
  __int64 v8; // rax
  unsigned int v9; // eax
  unsigned int v10; // [rsp+4h] [rbp-3Ch]
  __int64 *v11; // [rsp+8h] [rbp-38h]

  if ( a2[8] != a2[9] )
  {
    v2 = sub_1E80290(a1, (__int64)a2);
    v11 = (__int64 *)a2[12];
    if ( (__int64 *)a2[11] != v11 )
    {
      v10 = 0;
      v3 = (__int64 *)a2[11];
      v4 = 0;
      while ( 1 )
      {
        v5 = *v3;
        if ( !v2 )
          break;
        if ( v5 == **(_QWORD **)(v2 + 32) )
        {
LABEL_10:
          if ( v11 == ++v3 )
            return v4;
        }
        else
        {
          v6 = (_QWORD *)sub_1E80290(a1, *v3);
          if ( (_QWORD *)v2 != v6 )
          {
            while ( v6 )
            {
              v6 = (_QWORD *)*v6;
              if ( (_QWORD *)v2 == v6 )
                goto LABEL_13;
            }
            goto LABEL_10;
          }
LABEL_13:
          v8 = sub_1E80410(a1, v5);
          if ( !v8 )
            goto LABEL_10;
          v9 = *(_DWORD *)(v8 + 28);
          if ( v4 )
          {
            if ( v9 >= v10 )
              goto LABEL_10;
          }
          v10 = v9;
          v4 = v5;
          if ( v11 == ++v3 )
            return v4;
        }
      }
      sub_1E80290(a1, *v3);
      goto LABEL_13;
    }
  }
  return 0;
}
