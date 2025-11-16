// Function: sub_2ED3280
// Address: 0x2ed3280
//
void __fastcall sub_2ED3280(__int64 *a1, __int64 *a2, __int64 a3, _QWORD *a4)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned __int64 v7; // r12
  __int64 v8; // rax
  unsigned __int64 v9; // r15
  bool v10; // al
  __int64 v11; // r13
  __int64 *v12; // rdi
  __int64 v13; // r13
  __int64 v14; // r14
  unsigned int v15; // r12d
  __int64 *i; // r15
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // r8
  bool v22; // al
  __int64 *v23; // rdi
  __int64 v24; // r14
  unsigned int v25; // r12d
  unsigned __int64 v27; // [rsp+10h] [rbp-50h]
  __int64 *v28; // [rsp+18h] [rbp-48h]
  __int64 *v30; // [rsp+28h] [rbp-38h]

  if ( a1 != a2 )
  {
    v30 = a1 + 1;
    if ( a2 != a1 + 1 )
    {
      while ( 1 )
      {
        v12 = *(__int64 **)(a3 + 64);
        v13 = *a1;
        v14 = *v30;
        if ( !v12 )
          break;
        v5 = sub_2E39EA0(v12, *v30);
        v6 = *(_QWORD *)(a3 + 64);
        v7 = v5;
        if ( v6 )
        {
          v8 = sub_2E39EA0(*(__int64 **)(a3 + 64), v13);
          v6 = *(_QWORD *)(a3 + 64);
          v9 = v8;
        }
        else
        {
          v9 = 0;
        }
        if ( (unsigned __int8)sub_2EE68A0(*a4, *(_QWORD *)(a3 + 56), v6, 2) || !(v9 | v7) )
          goto LABEL_15;
        v10 = v7 < v9;
LABEL_9:
        v11 = *v30;
        if ( v10 )
        {
          if ( a1 != v30 )
            memmove(a1 + 1, a1, (char *)v30 - (char *)a1);
          *a1 = v11;
          if ( a2 == ++v30 )
            return;
        }
        else
        {
          for ( i = v30; ; i[1] = *i )
          {
            v23 = *(__int64 **)(a3 + 64);
            v24 = *(i - 1);
            v28 = i;
            if ( v23 )
            {
              v17 = sub_2E39EA0(v23, v11);
              v18 = *(_QWORD *)(a3 + 64);
              v19 = v17;
              if ( v18 )
              {
                v20 = sub_2E39EA0(*(__int64 **)(a3 + 64), v24);
                v18 = *(_QWORD *)(a3 + 64);
                v21 = v20;
              }
              else
              {
                v21 = 0;
              }
              v27 = v21;
              if ( !(unsigned __int8)sub_2EE68A0(*a4, *(_QWORD *)(a3 + 56), v18, 2) && v27 | v19 )
              {
                v22 = v19 < v27;
                goto LABEL_22;
              }
            }
            else
            {
              sub_2EE68A0(*a4, *(_QWORD *)(a3 + 56), 0, 2);
            }
            v25 = sub_2E5E7B0(*(_QWORD *)(a3 + 48), v11);
            v22 = v25 < (unsigned int)sub_2E5E7B0(*(_QWORD *)(a3 + 48), v24);
LABEL_22:
            --i;
            if ( !v22 )
              break;
          }
          *v28 = v11;
          if ( a2 == ++v30 )
            return;
        }
      }
      sub_2EE68A0(*a4, *(_QWORD *)(a3 + 56), 0, 2);
LABEL_15:
      v15 = sub_2E5E7B0(*(_QWORD *)(a3 + 48), v14);
      v10 = v15 < (unsigned int)sub_2E5E7B0(*(_QWORD *)(a3 + 48), v13);
      goto LABEL_9;
    }
  }
}
