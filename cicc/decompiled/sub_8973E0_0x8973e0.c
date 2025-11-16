// Function: sub_8973E0
// Address: 0x8973e0
//
void __fastcall sub_8973E0(__int64 **a1, __int64 a2, _QWORD *a3)
{
  __int64 *v3; // r14
  __int64 v5; // r13
  __int64 *v6; // rax
  __int64 **v7; // r12
  __int64 **v8; // rcx
  __int64 *v9; // r8
  __int64 *v10; // rax
  __int64 v11; // rbx
  _QWORD *v12; // rdx
  __int64 v13; // rdx
  __int64 **v14; // [rsp+0h] [rbp-40h]
  __int64 *v15; // [rsp+8h] [rbp-38h]

  v3 = *a1;
  if ( a3[2] && v3 )
  {
    v5 = 0;
    v6 = sub_896D70(0, a2, 1);
    v7 = 0;
    v8 = a1;
    v9 = v6;
    if ( (__int64)a3[2] <= 0 )
    {
LABEL_17:
      *v7 = v3;
    }
    else
    {
      while ( 1 )
      {
        v10 = v3;
        if ( *(_DWORD *)(*a3 + 4 * v5) )
        {
          v13 = *v9;
          if ( *v9 )
          {
            while ( (*(_BYTE *)(v13 + 24) & 8) != 0 )
            {
              v13 = *(_QWORD *)v13;
              if ( !v13 )
                goto LABEL_28;
            }
            v10 = v9;
            v9 = (__int64 *)v13;
          }
          else
          {
LABEL_28:
            v10 = v9;
            v9 = 0;
          }
        }
        if ( v7 )
          *v7 = v10;
        else
          *v8 = v10;
        do
        {
          v7 = (__int64 **)v10;
          v10 = (__int64 *)*v10;
        }
        while ( v10 && (v10[3] & 8) != 0 );
        v11 = *v3;
        if ( !*v3 )
          break;
        v12 = v3;
        while ( (*(_BYTE *)(v11 + 24) & 8) != 0 )
        {
          v12 = (_QWORD *)v11;
          if ( !*(_QWORD *)v11 )
            goto LABEL_20;
          v11 = *(_QWORD *)v11;
        }
        if ( *(_DWORD *)(*a3 + 4 * v5) )
        {
          *v12 = 0;
          v14 = v8;
          v15 = v9;
          sub_725130(v3);
          v8 = v14;
          v9 = v15;
        }
        ++v5;
        v3 = (__int64 *)v11;
        if ( a3[2] <= v5 )
          goto LABEL_17;
      }
      v11 = (__int64)v3;
LABEL_20:
      if ( *(_DWORD *)(*a3 + 4 * v5) )
      {
        *(_QWORD *)v11 = 0;
        sub_725130(v3);
      }
    }
  }
}
