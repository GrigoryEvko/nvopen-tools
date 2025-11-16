// Function: sub_3578DD0
// Address: 0x3578dd0
//
void __fastcall sub_3578DD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 i; // r12
  _QWORD *v10; // rax
  __int64 v11; // rcx
  __int64 **v12; // rbx
  __int64 **v13; // r14

  v4 = sub_2E5E6D0(*(_QWORD *)(a1 + 224), a2);
  v8 = 0;
  if ( v4 )
    v8 = *(unsigned int *)(v4 + 168);
  for ( i = a3; (unsigned int)v8 < *(_DWORD *)(i + 168); i = *(_QWORD *)i )
  {
    a3 = i;
    if ( !*(_QWORD *)i )
      goto LABEL_7;
  }
  i = a3;
LABEL_7:
  if ( !*(_BYTE *)(a1 + 620) )
    goto LABEL_14;
  v10 = *(_QWORD **)(a1 + 600);
  v5 = *(unsigned int *)(a1 + 612);
  v8 = (__int64)&v10[v5];
  if ( v10 == (_QWORD *)v8 )
  {
LABEL_13:
    if ( (unsigned int)v5 < *(_DWORD *)(a1 + 608) )
    {
      v11 = (unsigned int)(v5 + 1);
      *(_DWORD *)(a1 + 612) = v11;
      *(_QWORD *)v8 = i;
      ++*(_QWORD *)(a1 + 592);
LABEL_15:
      v12 = *(__int64 ***)(a1 + 752);
      v13 = &v12[*(unsigned int *)(a1 + 760)];
      if ( v13 == v12 )
      {
LABEL_21:
        sub_3578AC0(a1, i, v8, v11, v6, v7);
      }
      else
      {
        while ( !(unsigned __int8)sub_2E5E7F0(*v12, (__int64 *)i) )
        {
          if ( v13 == ++v12 )
            goto LABEL_21;
        }
      }
      return;
    }
LABEL_14:
    sub_C8CC70(a1 + 592, i, v8, v5, v6, v7);
    if ( !(_BYTE)v8 )
      return;
    goto LABEL_15;
  }
  while ( *v10 != i )
  {
    if ( (_QWORD *)v8 == ++v10 )
      goto LABEL_13;
  }
}
