// Function: .getrusage
// Address: 0x406dd0
//
// attributes: thunk
int getrusage(__rusage_who_t who, struct rusage *usage)
{
  return getrusage(who, usage);
}
