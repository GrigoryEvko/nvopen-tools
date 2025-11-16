// Function: sub_2EE8870
// Address: 0x2ee8870
//
__int64 __fastcall sub_2EE8870(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 *v3; // rdx
  __int64 v4; // r15
  __int64 *v5; // r14
  __int64 v6; // r13
  __int64 v7; // r12
  _QWORD *v8; // rax
  __int64 v10; // rax
  unsigned int v11; // eax
  unsigned int v12; // [rsp+4h] [rbp-3Ch]
  __int64 *v13; // [rsp+8h] [rbp-38h]

  if ( *(_DWORD *)(a2 + 120) )
  {
    v2 = sub_2EE86D0(a1, a2);
    v3 = *(__int64 **)(a2 + 112);
    v4 = v2;
    v13 = &v3[*(unsigned int *)(a2 + 120)];
    if ( v3 != v13 )
    {
      v12 = 0;
      v5 = *(__int64 **)(a2 + 112);
      v6 = 0;
      while ( 1 )
      {
        v7 = *v5;
        if ( !v4 )
          break;
        if ( v7 == **(_QWORD **)(v4 + 32) )
        {
LABEL_10:
          if ( v13 == ++v5 )
            return v6;
        }
        else
        {
          v8 = (_QWORD *)sub_2EE86D0(a1, *v5);
          if ( (_QWORD *)v4 != v8 )
          {
            while ( v8 )
            {
              v8 = (_QWORD *)*v8;
              if ( (_QWORD *)v4 == v8 )
                goto LABEL_13;
            }
            goto LABEL_10;
          }
LABEL_13:
          v10 = sub_2EE8840(a1, v7);
          if ( !v10 )
            goto LABEL_10;
          v11 = *(_DWORD *)(v10 + 28);
          if ( v6 )
          {
            if ( v11 >= v12 )
              goto LABEL_10;
          }
          v12 = v11;
          v6 = v7;
          if ( v13 == ++v5 )
            return v6;
        }
      }
      sub_2EE86D0(a1, *v5);
      goto LABEL_13;
    }
  }
  return 0;
}
